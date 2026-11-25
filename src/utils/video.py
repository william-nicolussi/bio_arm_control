import io
import os

import imageio.v2 as imageio
from PIL import Image


# ----- mp4 -----
def save_video(frames: list, path: str, fps: int = 30, hold_s: float = 1.0) -> None:
    """mp4 (H.264) at fps; the last frame is held for hold_s seconds, to read the plots.
    H.264 wants even sizes: the frames are cropped by one pixel if needed."""
    frames = list(frames) + [frames[-1]] * int(round(hold_s * fps))
    h, w = frames[0].shape[:2]
    frames = [f[:h - h % 2, :w - w % 2] for f in frames]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    print(f"Video saved in: {path} ({len(frames) / fps:.1f} s)")


# ----- player in the notebook -----
def frames_player(frames: list, t=None, fps: int = 30, width: int = None):
    """Player in the notebook: play / pause and a slider on the frames (like Manipulate in Mathematica).
    The frames are already rendered: moving the slider only changes the image shown. Needs ipywidgets.
    t (N,) [ms]: time of each frame, written next to the slider."""
    import ipywidgets as widgets            # only here: the rest of the module does not need it

    encoded = []
    for frame in frames:
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="png")
        encoded.append(buffer.getvalue())

    n = len(frames)
    image = widgets.Image(value=encoded[0], format="png", width=width or frames[0].shape[1])
    play = widgets.Play(value=0, min=0, max=n - 1, step=1, interval=int(round(1000 / fps)))
    slider = widgets.IntSlider(value=0, min=0, max=n - 1, description="frame",
                               layout=widgets.Layout(width="50%"))
    label = widgets.Label()
    widgets.jslink((play, "value"), (slider, "value"))   # play moves the slider, the slider moves the image

    def show(change):
        k = change["new"]
        image.value = encoded[k]
        label.value = "" if t is None else f"t = {t[k]:.0f} ms"

    slider.observe(show, names="value")
    show({"new": 0})
    return widgets.VBox([widgets.HBox([play, slider, label]), image])
