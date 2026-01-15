from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class ReencodeOptions:
    image_format: str  # "png" | "webp" | "jpeg"
    quality: int = 95  # used for lossy formats
    webp_lossless: bool = False


def reencode_dataset(in_root: Path, out_root: Path, opts: ReencodeOptions) -> None:
    in_root = in_root.resolve()
    out_root = out_root.resolve()

    if not (in_root / "manifest.json").exists():
        raise FileNotFoundError(f"Missing {(in_root / 'manifest.json')}")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text((in_root / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8")

    for split in ("train", "val", "test"):
        split_in = in_root / split
        if not split_in.exists():
            continue
        split_out = out_root / split
        split_out.mkdir(parents=True, exist_ok=True)
        for scene_in in sorted(p for p in split_in.iterdir() if p.is_dir()):
            scene_out = split_out / scene_in.name
            _reencode_scene(scene_in, scene_out, opts)


def _reencode_scene(scene_in: Path, scene_out: Path, opts: ReencodeOptions) -> None:
    scene_out.mkdir(parents=True, exist_ok=True)

    # Copy scene files that are not images.
    for name in ("meta.json", "gt_points.npz", "gt_charuco_corners.npz"):
        src = scene_in / name
        if src.exists():
            (scene_out / name).write_bytes(src.read_bytes())

    frames_in = scene_in / "frames.jsonl"
    if not frames_in.exists():
        raise FileNotFoundError(f"Missing {frames_in}")

    left_in = scene_in / "left"
    right_in = scene_in / "right"
    if not left_in.exists() or not right_in.exists():
        raise FileNotFoundError(f"Missing left/right dirs in {scene_in}")

    left_out = scene_out / "left"
    right_out = scene_out / "right"
    left_out.mkdir(exist_ok=True)
    right_out.mkdir(exist_ok=True)

    ext = _ext(opts)
    out_lines: list[str] = []
    for line in frames_in.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fr = json.loads(line)
        left_name = str(fr["left"])
        right_name = str(fr["right"])
        stem_l = Path(left_name).stem
        stem_r = Path(right_name).stem

        src_l = left_in / left_name
        src_r = right_in / right_name
        dst_l = left_out / f"{stem_l}.{ext}"
        dst_r = right_out / f"{stem_r}.{ext}"

        _reencode_image(src_l, dst_l, opts)
        _reencode_image(src_r, dst_r, opts)

        fr["left"] = dst_l.name
        fr["right"] = dst_r.name
        out_lines.append(json.dumps(fr))

    (scene_out / "frames.jsonl").write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _ext(opts: ReencodeOptions) -> str:
    fmt = opts.image_format.lower()
    if fmt == "png":
        return "png"
    if fmt == "webp":
        return "webp"
    if fmt in ("jpg", "jpeg"):
        return "jpg"
    raise ValueError("image_format must be png|webp|jpeg")


def _reencode_image(src: Path, dst: Path, opts: ReencodeOptions) -> None:
    fmt = opts.image_format.lower()
    q = int(opts.quality)
    q = max(0, min(100, q))
    with Image.open(src) as im:
        im = im.convert("L")
        if fmt == "png":
            im.save(dst)
            return
        if fmt == "webp":
            im.save(dst, lossless=bool(opts.webp_lossless), quality=q, method=6)
            return
        if fmt in ("jpg", "jpeg"):
            # Grayscale JPEG; keep high-frequency content as much as possible.
            im.save(dst, quality=q, optimize=True, progressive=False, subsampling=0)
            return
    raise ValueError("image_format must be png|webp|jpeg")

