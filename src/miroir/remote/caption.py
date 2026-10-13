# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false
"""Read an archive print's caption with a small open vision model. Runs only inside Modal.

Qwen3-VL Instruct is asked to copy the caption line that names the place
(`miroir.caption.CAPTION_QUESTION`), greedy, at most `CAPTION_TOKENS` tokens.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from collections.abc import Sequence

import torch
from PIL import Image as PILImage
from PIL import ImageOps

from miroir.caption import (
    CAPTION_MODELS,
    CAPTION_QUESTION,
    CAPTION_SIDE,
    CAPTION_TOKENS,
    clean_caption,
)
from miroir.eval.wire import CaptionChunk, CaptionPhoto

logger = logging.getLogger(__name__)

DEVICE = "cuda"


class CaptionReader:
    def __init__(self, model: str) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: PLC0415

        start = time.monotonic()
        name = CAPTION_MODELS[model]
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            name, dtype=torch.bfloat16, device_map=DEVICE
        ).eval()
        self.load_s = time.monotonic() - start
        logger.info("caption reader %s loaded in %.1f s", name, self.load_s)

    @torch.inference_mode()
    def read(self, jpeg: bytes) -> str:
        """The caption line of a photo, or "" when the model sees none."""
        with PILImage.open(io.BytesIO(jpeg)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((CAPTION_SIDE, CAPTION_SIDE), PILImage.Resampling.LANCZOS)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": CAPTION_QUESTION},
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
        out = self.model.generate(**inputs, max_new_tokens=CAPTION_TOKENS, do_sample=False)
        answer = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        return clean_caption(answer)


def read_captions(model: str, photos: Sequence[CaptionPhoto]) -> CaptionChunk:
    """Every photo's caption, and the time each took on the GPU."""
    reader = CaptionReader(model)
    reader.read(photos[0].jpeg)  # warm the kernels: the timings are of a warm reader
    texts: list[str] = []
    seconds: list[float] = []
    for photo in photos:
        tick = time.monotonic()
        texts.append(reader.read(photo.jpeg))
        torch.cuda.synchronize()
        seconds.append(time.monotonic() - tick)
    return CaptionChunk(
        ids=tuple(p.image_id for p in photos),
        texts=tuple(texts),
        seconds=tuple(seconds),
        load_s=reader.load_s,
    )


class BackgroundCaptions:
    """The site's caption reader, loaded in a thread beside the search models so it never
    delays a search; a print searched before it is ready waits for it at most `wait_s`."""

    def __init__(self, model: str, wait_s: float, *, start: bool = True) -> None:
        self._model = model
        self._wait_s = wait_s
        self._reader: CaptionReader | None = None
        self._ready = threading.Event()
        if start:
            threading.Thread(target=self._load, daemon=True).start()

    def load_now(self) -> None:
        """Load in the calling thread, once no other model is loading.

        `from_pretrained(dtype=bfloat16)` switches torch's default dtype for the whole
        process while it runs. Loaded beside EmbeddingGemma 2, it left that default at
        bfloat16, and every photo search then failed ("expected mat1 and mat2 to have the
        same dtype"). The search container therefore loads the reader after the other models.
        """
        self._load()

    def _load(self) -> None:
        try:
            self._reader = CaptionReader(self._model)
        except Exception:
            logger.exception("caption reader failed to load: prints are searched without it")
        finally:
            torch.set_default_dtype(torch.float32)
            self._ready.set()

    def read(self, jpeg: bytes) -> str:
        self._ready.wait(self._wait_s)
        return "" if self._reader is None else self._reader.read(jpeg)
