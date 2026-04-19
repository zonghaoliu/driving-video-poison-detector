"""Unified VLM backend. Selected by config.VLM_BACKEND.

Both backends expose the same two entry points so run_vlm.py and
gen_eval_steps.py don't care which model is answering:

  call_vlm(prompt, image_paths, schema) -> dict   # vision + JSON
  call_vlm_text(prompt)                -> str    # text only

OpenAI uses Structured Outputs (json_schema strict). Qwen has no server
side schema enforcement, so we inject the schema into the prompt and
extract the first balanced JSON object from the response. Parse failure
raises, which run_vlm's try/except turns into a skipped call + log line
(same behaviour as OpenAI timeouts today).

The Qwen model is loaded lazily on the first call and cached in-process.
"""
from __future__ import annotations
import base64
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import (
    VLM_BACKEND,
    OPENAI_API_KEY,
    MODEL,
    QWEN_MODEL_ID,
    QWEN_LOCAL_DIR,
    QWEN_MAX_NEW_TOKENS,
    QWEN_MIN_PIXELS,
    QWEN_MAX_PIXELS,
)


# ---------- OpenAI backend ----------

@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


def _b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def _openai_call(prompt: str, image_paths: list[Path], schema: dict | None) -> dict:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for p in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_b64(p)}", "detail": "low"},
        })
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "temperature": 0,
        "messages": [{"role": "user", "content": content}],
    }
    if schema:
        kwargs["response_format"] = {"type": "json_schema", "json_schema": schema}
    resp = _openai_client().with_options(timeout=90.0).chat.completions.create(**kwargs)
    txt = resp.choices[0].message.content
    return json.loads(txt) if schema else {"text": txt}


def _openai_text(prompt: str) -> str:
    resp = _openai_client().chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# ---------- Qwen backend ----------

_qwen_state: dict[str, Any] = {}


def _qwen_load() -> dict[str, Any]:
    if "model" in _qwen_state:
        return _qwen_state
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    ckpt = str(QWEN_LOCAL_DIR) if QWEN_LOCAL_DIR.exists() else QWEN_MODEL_ID
    print(f"[qwen] loading {ckpt} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        ckpt,
        min_pixels=QWEN_MIN_PIXELS,
        max_pixels=QWEN_MAX_PIXELS,
    )
    _qwen_state["model"] = model
    _qwen_state["processor"] = processor
    _qwen_state["torch"] = torch
    return _qwen_state


def _schema_text(schema: dict) -> str:
    body = schema.get("schema", schema)
    return json.dumps(body, indent=2, ensure_ascii=False)


def _extract_json(text: str) -> dict:
    """Pull the first top-level JSON object from a Qwen response."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"unbalanced JSON in response: {text[:200]!r}")


def _qwen_call(prompt: str, image_paths: list[Path], schema: dict | None) -> dict:
    state = _qwen_load()
    model = state["model"]
    processor = state["processor"]
    torch = state["torch"]
    from qwen_vl_utils import process_vision_info

    if schema:
        full_prompt = (
            f"{prompt}\n\n"
            "Respond with a SINGLE JSON object that conforms to the schema below. "
            "Output ONLY the JSON - no prose, no markdown fences, no comments.\n\n"
            f"Schema:\n{_schema_text(schema)}"
        )
    else:
        full_prompt = prompt

    content: list[dict[str, Any]] = [
        {"type": "image", "image": f"file://{Path(p).resolve()}"}
        for p in image_paths
    ]
    content.append({"type": "text", "text": full_prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=QWEN_MAX_NEW_TOKENS,
            do_sample=False,
        )
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
    out = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    return _extract_json(out) if schema else {"text": out}


def _qwen_text(prompt: str) -> str:
    return _qwen_call(prompt, image_paths=[], schema=None)["text"]


# ---------- Public interface ----------

def call_vlm(prompt: str, image_paths: list[Path], schema: dict) -> dict:
    """Run a VLM call with images; returns a dict matching the schema."""
    if VLM_BACKEND == "openai":
        return _openai_call(prompt, image_paths, schema)
    if VLM_BACKEND == "qwen":
        return _qwen_call(prompt, image_paths, schema)
    raise ValueError(f"unknown VLM_BACKEND: {VLM_BACKEND!r}")


def call_vlm_text(prompt: str) -> str:
    """Text-only VLM call (no images)."""
    if VLM_BACKEND == "openai":
        return _openai_text(prompt)
    if VLM_BACKEND == "qwen":
        return _qwen_text(prompt)
    raise ValueError(f"unknown VLM_BACKEND: {VLM_BACKEND!r}")
