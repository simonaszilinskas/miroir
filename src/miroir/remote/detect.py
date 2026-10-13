# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false
"""Object detectors for outlines (step C, F4). Runs only inside Modal containers.

Each detector takes a PIL photo and returns raw boxes (`miroir.outlines.Box`, in shares of
the photo) for step C's prompts: OWLv2 and Grounding DINO through transformers' zero-shot
detection, Qwen3-VL by asking for a JSON list of boxes. Raw boxes are kept down to the
option's `raw_threshold`; `miroir.outlines.pick_outlines` chooses what to show.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Protocol

import torch
from PIL import Image as PILImage

from miroir.outlines import DETECT_SIDE, PROMPTS, Box, DetectorKind, DetectorOption, clipped

logger = logging.getLogger(__name__)

DEVICE = "cuda"
#: Raw boxes kept per photo, surest first, after merging near-identical ones.
MAX_RAW = 80
#: Two raw boxes this alike (IoU) are one, whatever their prompts.
SAME_BOX = 0.85
QWEN_MAX_TOKENS = 2048
QWEN_QUESTION = (
    "Outline every artwork in this photo: each painting, tapestry, drawing, sculpture, statue, "
    "vase, piece of furniture, building and stained glass window, one box per object. Answer "
    'with a JSON list only: [{"bbox_2d": [x1, y1, x2, y2], "label": "painting"}].'
)


class Detector(Protocol):
    def detect(self, image: PILImage.Image) -> list[Box]: ...


def detect_size(image: PILImage.Image) -> PILImage.Image:
    """The photo at most DETECT_SIDE on its long side, as the page sends it."""
    out = image.convert("RGB")
    out.thumbnail((DETECT_SIDE, DETECT_SIDE), PILImage.Resampling.LANCZOS)
    return out


def _merge(boxes: list[Box]) -> list[Box]:
    from miroir.outlines import iou  # noqa: PLC0415

    kept: list[Box] = []
    for box in sorted(boxes, key=lambda b: -b.score):
        if len(kept) >= MAX_RAW:
            break
        if all(iou(box, k) <= SAME_BOX for k in kept):
            kept.append(box)
    return kept


class Owlv2Detector:
    def __init__(self, option: DetectorOption) -> None:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor  # noqa: PLC0415

        self.option = option
        self.processor = Owlv2Processor.from_pretrained(option.model)
        self.model = Owlv2ForObjectDetection.from_pretrained(option.model).eval().to(DEVICE)
        self.labels = [list(PROMPTS)]

    @torch.inference_mode()
    def detect(self, image: PILImage.Image) -> list[Box]:
        inputs = self.processor(text=self.labels, images=image, return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs)
        # OWLv2 pads the photo to a square at the bottom and right: boxes are in that square.
        side = max(image.width, image.height)
        found = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=torch.tensor([(side, side)]),
            threshold=self.option.raw_threshold,
            text_labels=self.labels,
        )[0]
        return _merge(_boxes(found["boxes"], found["scores"], found["text_labels"], image))


class GroundingDinoDetector:
    def __init__(self, option: DetectorOption) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # noqa: PLC0415

        self.option = option
        self.processor = AutoProcessor.from_pretrained(option.model)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(option.model).eval().to(DEVICE)
        )
        self.labels = [list(PROMPTS)]

    @torch.inference_mode()
    def detect(self, image: PILImage.Image) -> list[Box]:
        inputs = self.processor(images=image, text=self.labels, return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs)
        found = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.option.raw_threshold,
            text_threshold=self.option.raw_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        labels = found.get("text_labels", found.get("labels"))
        return _merge(_boxes(found["boxes"], found["scores"], labels, image))


class Qwen3VlDetector:
    """Asks Qwen3-VL for boxes; it answers in thousandths of the photo. No scores: the order
    of its answer stands in for one (first = 1.0)."""

    def __init__(self, option: DetectorOption) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: PLC0415

        self.option = option
        self.processor = AutoProcessor.from_pretrained(option.model)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            option.model, dtype=torch.bfloat16, device_map=DEVICE
        ).eval()

    @torch.inference_mode()
    def detect(self, image: PILImage.Image) -> list[Box]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": QWEN_QUESTION},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(DEVICE)
        out = self.model.generate(**inputs, max_new_tokens=QWEN_MAX_TOKENS, do_sample=False)
        answer = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        return _merge(parse_qwen_boxes(answer))


def parse_qwen_boxes(answer: str) -> list[Box]:
    """Boxes from Qwen3-VL's JSON answer; tolerates code fences and a cut-off last item."""
    text = answer.replace("```json", "").replace("```", "").strip()
    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        items = [
            {"bbox_2d": [float(v) for v in m.groups()], "label": ""}
            for m in re.finditer(
                r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", text
            )
        ]
    boxes: list[Box] = []
    count = len(items) if isinstance(items, list) else 0
    for n, item in enumerate(items if isinstance(items, list) else []):
        coords = item.get("bbox_2d") if isinstance(item, dict) else None
        if not isinstance(coords, list) or len(coords) != 4:
            continue
        x0, y0, x1, y1 = (float(c) / 1000 for c in coords)
        box = clipped(
            x0, y0, x1, y1, score=1.0 - n / max(count, 1), label=str(item.get("label", ""))
        )
        if box is not None:
            boxes.append(box)
    return boxes


def _boxes(boxes, scores, labels, image: PILImage.Image) -> list[Box]:
    out: list[Box] = []
    w, h = image.width, image.height
    for (x0, y0, x1, y1), score, label in zip(boxes.tolist(), scores.tolist(), labels, strict=True):
        box = clipped(x0 / w, y0 / h, x1 / w, y1 / h, score=float(score), label=str(label))
        if box is not None:
            out.append(box)
    return out


_LOADERS = {
    DetectorKind.OWLV2: Owlv2Detector,
    DetectorKind.GROUNDING_DINO: GroundingDinoDetector,
    DetectorKind.QWEN3_VL: Qwen3VlDetector,
}


def load_detector(option: DetectorOption) -> Detector:
    start = time.monotonic()
    loaded = _LOADERS[option.kind](option)
    logger.info("detector %s loaded in %.1f s", option.model, time.monotonic() - start)
    return loaded
