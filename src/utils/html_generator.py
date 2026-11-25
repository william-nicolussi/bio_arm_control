import base64
import io
import json
import os

from PIL import Image

TEMPLATE = os.path.join(os.path.dirname(__file__), "player_template.html")


def save_player_html(layers, path: str, t=None, fps: int = 30, title: str = "", quality: int = 85) -> None:
    """layers: {name: frames} (e.g. {"top": [...], "side": [...], "plots": [...]}), every list with the same
    number of RGB images (H, W, 3); a plain list of frames is one layer. t (N,) [ms]: time of each frame,
    shown next to the slider. quality: JPEG quality 0-100 (about 50-100 kB per image)."""
    layers = layers if isinstance(layers, dict) else {"frames": layers}
    lengths = {name: len(frames) for name, frames in layers.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"every layer needs the same number of frames: {lengths}")
    with open(TEMPLATE, encoding="utf-8") as f:
        page = f.read()
    images = {name: [_jpeg(frame, quality) for frame in frames] for name, frames in layers.items()}
    times = [] if t is None else [float(x) for x in t]
    page = (page.replace("__TITLE__", title)
                .replace("__LAYERS__", json.dumps(images))
                .replace("__TIMES__", json.dumps(times))
                .replace("__INTERVAL__", str(int(round(1000 / fps)))))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    n = next(iter(lengths.values()))
    print(f"Player saved in: {path} ({len(layers)} layers x {n} frames, {os.path.getsize(path) / 1e6:.1f} MB)")


def _jpeg(frame, quality: int) -> str:
    """One frame as a data URL that an <img> can show."""
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="jpeg", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
