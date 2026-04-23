"""
LoRA Fine-tuning for Qwen2.5-VL - Optimized for run_vlm.py two-stage pipeline.

This module fine-tunes Qwen2.5-VL specifically for the two-stage evaluation:
  - Sub-task A: Per-frame semantic + cross-view scoring
  - Sub-task B: Per-view temporal, logical + decision scoring

The strategy:
1. Load hand-annotated video-level ground truth
2. Collect corresponding sub-task outputs (frame-level and view-level)
3. Train separate LoRA adapters for each sub-task
4. Each adapter learns to refine scores based on the observations

Usage:
    trainer_a = QwenLoRATwoStageTrainer(sub_task="A")
    trainer_a.prepare_training_data(...)
    trainer_a.train(...)
    trainer_a.save_adapter("outputs/lora_semantic")

    trainer_b = QwenLoRATwoStageTrainer(sub_task="B")
    trainer_b.prepare_training_data(...)
    trainer_b.train(...)
    trainer_b.save_adapter("outputs/lora_temporal")
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

try:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, AutoModelForCausalLM
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

logger = logging.getLogger(__name__)


class SubTaskDataset(Dataset):
    """Dataset for a single sub-task with multiple calls per video."""

    def __init__(
        self,
        video_ids: list[str],
        sub_task_outputs: list[dict],
        ground_truth_scores: list[dict],
        sub_task: str = "A",
    ):
        """Initialize dataset.

        Args:
            video_ids: List of video IDs
            sub_task_outputs: Per-frame (A) or per-view (B) outputs from run_vlm.py
            ground_truth_scores: Hand-annotated scores at video level
            sub_task: "A" for semantic or "B" for temporal/logical
        """
        self.video_ids = video_ids
        self.sub_task_outputs = sub_task_outputs
        self.ground_truth_scores = ground_truth_scores
        self.sub_task = sub_task

    def __len__(self) -> int:
        # Multiple samples per video (8 for sub-task A, 3 for sub-task B)
        return len(self.sub_task_outputs)

    def __getitem__(self, idx: int) -> dict:
        """Get a single sub-task evaluation sample."""
        output = self.sub_task_outputs[idx]

        # Find corresponding video
        video_id = output.get("video_id", "")
        gt = next((g for g in self.ground_truth_scores if g["video_id"] == video_id), None)

        if not gt:
            raise ValueError(f"No ground truth for video {video_id}")

        # Format input text from observations
        if self.sub_task == "A":
            input_text = self._format_semantic_input(output)
            target_text = self._format_semantic_target(gt, output)
        else:  # sub_task == "B"
            input_text = self._format_temporal_input(output)
            target_text = self._format_temporal_target(gt, output)

        return {
            "video_id": video_id,
            "input_text": input_text,
            "target_text": target_text,
            "output": output,
            "ground_truth": gt,
        }

    @staticmethod
    def _format_semantic_input(output: dict) -> str:
        """Format Sub-task A input from observations."""
        obs = output.get("per_view_observations", [])
        obs_text = "\n".join(
            f"- {o.get('view')}: semantic='{o.get('semantic_issue')}', "
            f"cross_view='{o.get('cross_view_issue')}'"
            for o in obs
        )
        return f"""Per-frame semantic evaluation (frame {output.get('frame_idx')}):
{obs_text}

Overall finding: {output.get('overall_finding')}

Based on these observations, provide the semantic and cross-view scores."""

    @staticmethod
    def _format_semantic_target(gt: dict, output: dict) -> str:
        """Format Sub-task A target (ground truth scores)."""
        return f"""semantic_score: {gt['scores']['semantic']:.1f}
cross_view_score: {gt['scores']['semantic']:.1f}
reasoning: {gt['reasoning']}"""

    @staticmethod
    def _format_temporal_input(output: dict) -> str:
        """Format Sub-task B input from observations."""
        obs = output.get("temporal_observations", [])
        obs_text = "\n".join(
            f"- Frames {o.get('frame_range')}: logical='{o.get('logical_issue')}', "
            f"decision='{o.get('decision_issue')}'"
            for o in obs
        )
        return f"""Per-view temporal evaluation ({output.get('view')}):
{obs_text}

Overall finding: {output.get('overall_finding')}

Based on these observations, provide the logical and decision scores."""

    @staticmethod
    def _format_temporal_target(gt: dict, output: dict) -> str:
        """Format Sub-task B target (ground truth scores)."""
        return f"""logical_score: {gt['scores']['logical']:.1f}
decision_score: {gt['scores']['decision']:.1f}
reasoning: {gt['reasoning']}"""


class QwenLoRATwoStageTrainer:
    """LoRA fine-tuner optimized for run_vlm.py two-stage pipeline."""

    def __init__(
        self,
        sub_task: str = "A",
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lora_rank: int = 8,
        lora_alpha: int = 16,
    ):
        """Initialize trainer for specific sub-task.

        Args:
            sub_task: "A" (semantic) or "B" (temporal/logical)
            model_id: HuggingFace model ID
            device: Device to train on
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha scaling
        """
        if not HAS_PEFT:
            raise ImportError("Please install peft: pip install peft")

        self.sub_task = sub_task
        self.device = torch.device(device)
        self.model_id = model_id

        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        logger.info(f"Loading model {model_id} for Sub-task {sub_task}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=str(self.device),
            torch_dtype=torch.bfloat16,
        )

        self.model = get_peft_model(self.base_model, self.lora_config)
        logger.info(f"Applied LoRA for Sub-task {sub_task}")
        self.model.print_trainable_parameters()

        self.optimizer = None
        self.training_history = {"loss": [], "val_loss": []}

    def prepare_training_data(
        self,
        vlm_raw_dir: Path,
        hand_annotated_json: Path,
        split_ratio: float = 0.8,
    ) -> tuple[DataLoader, DataLoader]:
        """Prepare training data from run_vlm.py outputs and ground truth.

        Args:
            vlm_raw_dir: Directory with outputs/exp_*/vlm_raw/*.json from run_vlm.py
            hand_annotated_json: Path to merged_output_raw.json
            split_ratio: Train/val split

        Returns:
            (train_dataloader, val_dataloader)
        """
        logger.info(f"Loading Sub-task {self.sub_task} outputs from {vlm_raw_dir}")

        # Load hand-annotated ground truth
        with open(hand_annotated_json) as f:
            ground_truth = json.load(f)

        gt_dict = {str(r["video_id"]): r for r in ground_truth}

        # Load VLM raw outputs
        sub_task_outputs = []
        raw_files = sorted(Path(vlm_raw_dir).glob("*.json"))

        for raw_file in raw_files:
            with open(raw_file) as f:
                raw = json.load(f)

            video_id = raw_file.stem

            # Extract appropriate sub-task outputs
            if self.sub_task == "A":
                for item in raw.get("semantic", []):
                    item["video_id"] = video_id
                    sub_task_outputs.append(item)
            else:  # sub_task == "B"
                for item in raw.get("temporal", []):
                    item["video_id"] = video_id
                    sub_task_outputs.append(item)

        logger.info(f"Found {len(sub_task_outputs)} Sub-task {self.sub_task} outputs")

        # Filter to only videos with ground truth
        filtered = [
            o for o in sub_task_outputs
            if o.get("video_id") in gt_dict
        ]
        logger.info(f"Filtered to {len(filtered)} samples with ground truth")

        # Split
        num_train = int(len(filtered) * split_ratio)
        train_outputs = filtered[:num_train]
        val_outputs = filtered[num_train:]

        train_video_ids = list(set(o.get("video_id") for o in train_outputs))
        val_video_ids = list(set(o.get("video_id") for o in val_outputs))
        train_gts = [gt_dict[vid] for vid in train_video_ids]
        val_gts = [gt_dict[vid] for vid in val_video_ids]

        logger.info(f"Train: {len(train_outputs)} samples from {len(train_video_ids)} videos")
        logger.info(f"Val: {len(val_outputs)} samples from {len(val_video_ids)} videos")

        # Create datasets
        train_dataset = SubTaskDataset(
            train_video_ids, train_outputs, train_gts, self.sub_task
        )
        val_dataset = SubTaskDataset(
            val_video_ids, val_outputs, val_gts, self.sub_task
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        return train_loader, val_loader

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 3,
        learning_rate: float = 1e-4,
        warmup_steps: int = 50,
    ) -> dict:
        """Train LoRA adapter on sub-task outputs.

        Args:
            train_loader: Training dataloader
            val_loader: Optional validation dataloader
            epochs: Number of epochs
            learning_rate: Adam learning rate
            warmup_steps: Warmup steps

        Returns:
            Training history
        """
        self.model.train()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        from torch.optim.lr_scheduler import LinearLR
        scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_steps,
        )

        for epoch in range(epochs):
            logger.info(f"\n=== Sub-task {self.sub_task} Epoch {epoch + 1}/{epochs} ===")
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                try:
                    input_text = batch["input_text"][0]
                    target_text = batch["target_text"][0]

                    # Prepare input
                    full_text = f"{input_text}\n\nTarget:\n{target_text}"

                    inputs = self.processor(
                        text=full_text,
                        return_tensors="pt",
                    ).to(self.device)

                    # Forward pass
                    outputs = self.model(**inputs)
                    loss = outputs.loss

                    # Backward pass
                    loss.backward()
                    epoch_loss += loss.item()

                    if (batch_idx + 1) % 4 == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        scheduler.step()
                        self.optimizer.zero_grad()

                    if (batch_idx + 1) % 5 == 0:
                        avg_loss = epoch_loss / (batch_idx + 1)
                        logger.info(
                            f"Batch {batch_idx + 1}/{len(train_loader)} - "
                            f"Loss: {avg_loss:.4f}"
                        )

                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    continue

            avg_epoch_loss = epoch_loss / len(train_loader)
            self.training_history["loss"].append(avg_epoch_loss)
            logger.info(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

            # Validation
            if val_loader:
                val_loss = self._validate(val_loader)
                self.training_history["val_loss"].append(val_loss)
                logger.info(f"Validation loss: {val_loss:.4f}")

        return self.training_history

    def _validate(self, val_loader: DataLoader) -> float:
        """Compute validation loss."""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                try:
                    input_text = batch["input_text"][0]
                    target_text = batch["target_text"][0]
                    full_text = f"{input_text}\n\nTarget:\n{target_text}"

                    inputs = self.processor(
                        text=full_text,
                        return_tensors="pt",
                    ).to(self.device)

                    outputs = self.model(**inputs)
                    val_loss += outputs.loss.item()
                    num_batches += 1

                except Exception as e:
                    logger.error(f"Validation error: {e}")
                    continue

        self.model.train()
        return val_loss / max(num_batches, 1)

    def save_adapter(self, save_path: Path | str) -> None:
        """Save LoRA adapter weights."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        logger.info(f"Saved LoRA Sub-task {self.sub_task} adapter to {save_path}")

    def load_adapter(self, adapter_path: Path | str) -> None:
        """Load pre-trained LoRA adapter."""
        from peft import PeftModel
        adapter_path = Path(adapter_path)
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        logger.info(f"Loaded LoRA adapter from {adapter_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("QwenLoRATwoStageTrainer module loaded.")
    print("Use for training Sub-task A (semantic) or Sub-task B (temporal).")
    print("\nExample:")
    print("  trainer_a = QwenLoRATwoStageTrainer(sub_task='A')")
    print("  train_loader, val_loader = trainer_a.prepare_training_data(...)")
    print("  trainer_a.train(train_loader, val_loader)")
    print("  trainer_a.save_adapter('outputs/lora_semantic')")
