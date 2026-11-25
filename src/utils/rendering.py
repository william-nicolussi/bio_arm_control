from contextlib import contextmanager

import mujoco
import numpy as np

# tendon colours: controlled muscles from REST (activation 0) to ACTIVE (activation 1), the others OTHER.
# MyoSuite draws every tendon red: without this the activation would not be visible.
REST_RGBA = np.array([0.75, 0.78, 0.9, 1.0], dtype=np.float32)
ACTIVE_RGBA = np.array([0.85, 0.05, 0.05, 1.0], dtype=np.float32)
OTHER_RGBA = np.array([0.8, 0.8, 0.8, 0.25], dtype=np.float32)

# rendering quality: shadows (4096 px shadow map in MyoSuite), floor reflections, antialiasing samples.
# Measured on myoarm (480x480, software OpenGL): full ~1000 ms per image, normal ~270 ms, fast ~105 ms.
QUALITY = {"fast": dict(shadows=False, reflections=False, offsamples=0),
           "normal": dict(shadows=False, reflections=False, offsamples=4),
           "full": dict(shadows=True, reflections=True, offsamples=4)}


# ----- cameras -----
def free_camera(lookat, distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
    """Free camera looking at lookat [m] from distance [m]; azimuth and elevation [deg] as in the MuJoCo viewer
    (azimuth 0 = looking along +x, elevation -90 = looking down)."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


VIEWS = ("top", "front", "side")


def view_camera(view: str, lookat, forward_xy) -> mujoco.MjvCamera:
    """The views of a subject whose forward direction is forward_xy (world x, y), looking at lookat [m]:
    "top" from above, with the forward direction up in the image; "front" in front of the subject, a bit from
    above, looking back at it; "side" from the right of the subject, at the height of lookat (heights and
    inclinations, e.g. of the humerus, are seen without foreshortening)."""
    azimuth = float(np.rad2deg(np.arctan2(forward_xy[1], forward_xy[0])))
    if view == "top":
        return free_camera(lookat, distance=1.4, azimuth=azimuth, elevation=-89.0)
    if view == "front":
        return free_camera(lookat, distance=1.6, azimuth=azimuth + 180.0, elevation=-20.0)
    if view == "side":
        return free_camera(lookat, distance=1.6, azimuth=azimuth + 90.0, elevation=-3.0)
    raise ValueError(f"view must be one of {VIEWS}, got '{view}'")


# ----- MuJoCo images -----
def render_frames(model: mujoco.MjModel, qpos, camera, height: int = 480, width: int = 480,
                  act=None, tendons=None, markers: list = None, quality: str = "fast") -> list:
    """One RGB image (height, width, 3) per row of qpos (N, nq): only the position stage, no dynamics.
    camera: an MjvCamera (free_camera, view_camera), the name of a camera of the model, or -1 (default camera).
    act (N, M) with tendons (M,) tendon ids: the tendon of each muscle goes from light blue to red with its
    activation; the other tendons are drawn light grey.
    markers: [(positions (N, 3) or (3,), radius [m], rgba), ...]: one sphere per frame, drawn in the scene,
    not in the model. quality: "fast", "normal" or "full" (QUALITY)."""
    qpos = np.asarray(qpos, dtype=float)
    data = mujoco.MjData(model)
    frames = []
    with _renderer(model, height, width, quality, grey_tendons=act is not None) as (renderer, options):
        for k in range(len(qpos)):
            data.qpos[:] = qpos[k]
            mujoco.mj_fwdPosition(model, data)          # kinematics + tendon paths (wrapping)
            if act is not None:
                a = np.clip(np.asarray(act[k], dtype=np.float32), 0.0, 1.0)[:, None]
                model.tendon_rgba[tendons] = (1.0 - a) * REST_RGBA + a * ACTIVE_RGBA
            renderer.update_scene(data, camera)
            _apply_quality(renderer.scene, options)
            for positions, radius, rgba in markers or []:
                positions = np.asarray(positions, dtype=float)
                _add_sphere(renderer.scene, positions[k] if positions.ndim == 2 else positions, radius, rgba)
            frames.append(renderer.render().copy())
    return frames


def render_postures(model: mujoco.MjModel, qpos, camera, height: int = 480, width: int = 480,
                    markers: list = None, ghost_alpha: float = 0.2, quality: str = "normal") -> np.ndarray:
    """One RGB image with several postures (rows of qpos): the last one solid, the others as transparent ghosts
    (only the bodies that move). The tendons are light grey: here they carry no activation.
    markers: [(positions (3,) or (K, 3), radius [m], rgba), ...]: every position is drawn (e.g. a path)."""
    qpos = np.atleast_2d(np.asarray(qpos, dtype=float))
    data, ghost = mujoco.MjData(model), mujoco.MjData(model)
    with _renderer(model, height, width, quality, grey_tendons=True) as (renderer, options):
        data.qpos[:] = qpos[-1]
        mujoco.mj_fwdPosition(model, data)
        renderer.update_scene(data, camera)
        scene = renderer.scene
        _apply_quality(scene, options)
        option, perturb = mujoco.MjvOption(), mujoco.MjvPerturb()
        for q in qpos[:-1]:
            ghost.qpos[:] = q
            mujoco.mj_fwdPosition(model, ghost)
            first = scene.ngeom
            mujoco.mjv_addGeoms(model, ghost, option, perturb, mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
            for g in range(first, scene.ngeom):
                scene.geoms[g].rgba[3] *= ghost_alpha
        for positions, radius, rgba in markers or []:
            for pos in np.atleast_2d(np.asarray(positions, dtype=float)):
                _add_sphere(scene, pos, radius, rgba)
        return renderer.render().copy()


@contextmanager
def _renderer(model: mujoco.MjModel, height: int, width: int, quality: str, grey_tendons: bool):
    """Offscreen renderer with the antialiasing of quality; the model (the one of the simulation) is changed
    only while rendering: antialiasing, buffer size and tendon colours are restored at the end."""
    if quality not in QUALITY:
        raise ValueError(f"quality must be one of {list(QUALITY)}, got '{quality}'")
    options = QUALITY[quality]
    rgba0, offsamples0 = model.tendon_rgba.copy(), model.vis.quality.offsamples
    model.vis.quality.offsamples = options["offsamples"]                    # read when the renderer is created
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)      # size of the offscreen buffer
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    if grey_tendons:
        model.tendon_rgba[:] = OTHER_RGBA
    try:
        with mujoco.Renderer(model, height, width) as renderer:
            yield renderer, options
    finally:
        model.tendon_rgba[:] = rgba0
        model.vis.quality.offsamples = offsamples0


def _apply_quality(scene: mujoco.MjvScene, options: dict) -> None:
    """Shadows and reflections: they cost most of the rendering time."""
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = options["shadows"]
    scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = options["reflections"]


def _add_sphere(scene: mujoco.MjvScene, pos, radius: float, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0.0, 0.0]), np.asarray(pos, dtype=float),
                        np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def side_by_side(images: list) -> np.ndarray:
    """Images (H_i, W_i, 3) in a row, centred in height on a white background."""
    height = max(image.shape[0] for image in images)
    row = np.full((height, sum(image.shape[1] for image in images), 3), 255, dtype=np.uint8)
    x = 0
    for image in images:
        top = (height - image.shape[0]) // 2
        row[top:top + image.shape[0], x:x + image.shape[1]] = image
        x += image.shape[1]
    return row
