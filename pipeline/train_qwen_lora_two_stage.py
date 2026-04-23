"""
Training script for two-stage LoRA fine-tuning (optimized for run_vlm.py).

This trains separate LoRA adapters for:
  - Sub-task A: Semantic + cross-view scoring
  - Sub-task B: Temporal + logical + decision scoring

Usage:
    python train_qwen_lora_two_stage.py --sub-task A
    python train_qwen_lora_two_stage.py --sub-task B
    python train_qwen_lora_two_stage.py --both  # Train both
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import torch

from config import OUTPUT_DIR, HUMAN_JSON, QWEN_MODEL_ID
from qwen_lora_two_stage import QwenLoRATwoStageTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_sub_task(
    sub_task: str,
    vlm_raw_dir: Path,
    hand_annotated_path: Path,
    save_dir: Path,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    lora_rank: int = 8,
):
    """Train LoRA adapter for a single sub-task.

    Args:
        sub_task: "A" (semantic) or "B" (temporal)
        vlm_raw_dir: Directory with run_vlm.py outputs
        hand_annotated_path: Path to hand-annotated JSON
        save_dir: Directory to save adapter
        epochs: Training epochs
        learning_rate: Adam learning rate
        lora_rank: LoRA rank
    """
    logger.info("=" * 70)
    logger.info(f"Training Sub-task {sub_task} LoRA Adapter")
    logger.info("=" * 70)
    logger.info(f"Model: {QWEN_MODEL_ID}")
    logger.info(f"VLM outputs: {vlm_raw_dir}")
    logger.info(f"Ground truth: {hand_annotated_path}")
    logger.info(f"Save to: {save_dir}")
    logger.info(f"Epochs: {epochs}, LR: {learning_rate:.2e}, Rank: {lora_rank}")
    logger.info("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Initialize trainer
    trainer = QwenLoRATwoStageTrainer(
        sub_task=sub_task,
        device=device,
        lora_rank=lora_rank,
    )

    # Prepare data
    logger.info(f"\nPreparing Sub-task {sub_task} training data...")
    train_loader, val_loader = trainer.prepare_training_data(
        vlm_raw_dir=vlm_raw_dir,
        hand_annotated_json=hand_annotated_path,
        split_ratio=0.8,
    )

    # Train
    logger.info(f"\nTraining Sub-task {sub_task}...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    # Save
    adapter_path = save_dir / f"lora_subtask_{sub_task}"
    trainer.save_adapter(adapter_path)

    # Summary
    summary = {
        "sub_task": sub_task,
        "model": QWEN_MODEL_ID,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "device": device,
        "final_train_loss": history["loss"][-1] if history["loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
    }

    summary_path = save_dir / f"summary_subtask_{sub_task}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSub-task {sub_task} training complete!")
    logger.info(f"Adapter: {adapter_path}")
    logger.info(f"Summary: {summary_path}")

    return trainer, history


def main(
    sub_task: str = "both",
    vlm_raw_dir: Path = None,
    hand_annotated_path: Path = None,
    save_dir: Path = None,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    lora_rank: int = 8,
):
    """Train LoRA adapters for run_vlm.py pipeline.

    Args:
        sub_task: "A", "B", or "both"
        vlm_raw_dir: Directory with VLM raw outputs
        hand_annotated_path: Path to ground truth JSON
        save_dir: Directory to save adapters
        epochs: Training epochs
        learning_rate: Learning rate
        lora_rank: LoRA rank
    """
    # Set defaults
    vlm_raw_dir = vlm_raw_dir or OUTPUT_DIR / "exp_criteria_2" / "vlm_raw"
    hand_annotated_path = hand_annotated_path or Path(HUMAN_JSON)
    save_dir = save_dir or OUTPUT_DIR / "qwen_lora_two_stage"

    # Validate
    if not vlm_raw_dir.exists():
        logger.error(f"VLM raw directory not found: {vlm_raw_dir}")
        logger.info("Run run_vlm.py first to generate VLM outputs")
        return

    if not hand_annotated_path.exists():
        logger.error(f"Hand-annotated file not found: {hand_annotated_path}")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    # Train sub-task(s)
    if sub_task in ("A", "both"):
        train_sub_task(
            "A",
            vlm_raw_dir,
            hand_annotated_path,
            save_dir,
            epochs,
            learning_rate,
            lora_rank,
        )

    if sub_task in ("B", "both"):
        train_sub_task(
            "B",
            vlm_raw_dir,
            hand_annotated_path,
            save_dir,
            epochs,
            learning_rate,
            lora_rank,
        )

    logger.info("\n" + "=" * 70)
    logger.info("Training complete!")
    logger.info(f"Adapters saved to: {save_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Use trained adapters with python infer_qwen_lora_two_stage.py")
    logger.info("2. Compare improved predictions with evaluate.py")
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LoRA adapters for two-stage run_vlm.py pipeline"
    )
    parser.add_argument(
        "--sub-task",
        choices=["A", "B", "both"],
        default="both",
        help="Which sub-task to train",
    )
    parser.add_argument(
        "--vlm-raw-dir",
        type=Path,
        help="Directory with run_vlm.py outputs (vlm_raw/)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Path to hand-annotated JSON",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Directory to save adapters",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank",
    )

    args = parser.parse_args()

    main(
        sub_task=args.sub_task,
        vlm_raw_dir=args.vlm_raw_dir,
        hand_annotated_path=args.data_path,
        save_dir=args.save_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
    )
