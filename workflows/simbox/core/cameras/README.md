# Cameras

Template-based cameras for simbox tasks. All cameras currently use a single generic implementation, `CustomCamera`, which is configured entirely from the task YAML.

## Available cameras

| Camera class    | Notes |
|-----------------|-------|
| `CustomCamera`  | Generic pinhole RGB-D camera with configurable intrinsics and pose. |

Importing `CustomCamera` in your task (e.g. `banana.py`) is enough to register it via `@register_camera`.

---

## Customizing a camera configuration

Camera behavior is controlled by the config (`cfg`) passed into `CustomCamera.__init__` in `banana.py`. You typically edit the YAML under `configs/simbox/...`.

### 1. Top-level camera fields

Each camera entry in the YAML should provide:

- **`name`**: Unique camera name (string). Used for prim paths and as the key in `task.cameras`.
- **`parent`**: Optional prim path (under the task root) that the camera mount is attached to. Empty string (`""`) means no specific parent.
- **`translation`**: Initial camera translation in world or parent frame, as a list of three floats `[x, y, z]` (meters).
- **`orientation`**: Initial camera orientation as a quaternion `[w, x, y, z]`.
- **`camera_axes`**: Axes convention for `set_local_pose` (e.g. `[1.0, 0.0, 0.0]` etc. – follow existing configs).

These values are used in `banana.py` when calling:

```python
camera.set_local_pose(
    translation=cfg["translation"],
    orientation=cfg["orientation"],
    camera_axes=cfg["camera_axes"],
)
```

### 2. Required `params` fields

Inside each camera config there is a `params` dict that controls the optics and intrinsics. `CustomCamera` expects:

- **`pixel_size`** (`float`, microns)
  Physical pixel size on the sensor. Used to compute horizontal/vertical aperture and focal length.

- **`f_number`** (`float`)
  Lens f-number. Used in `set_lens_aperture(f_number * 100.0)`.

- **`focus_distance`** (`float`, meters)
  Focus distance passed to `set_focus_distance`.

- **`camera_params`** (`[fx, fy, cx, cy]`)
  Intrinsic matrix parameters in pixel units:
  - `fx`, `fy`: focal lengths in x/y (pixels)
  - `cx`, `cy`: principal point (pixels)

- **`resolution_width`** (`int`)
  Image width in pixels.

- **`resolution_height`** (`int`)
  Image height in pixels.

Optional:

- **`output_mode`** (`"rgb"` or `"diffuse_albedo"`, default `"rgb"`)
  Controls which color source is used in `get_observations()`.

### 3. How the parameters are used in `CustomCamera`

Given `cfg["params"]`, `CustomCamera` does the following:

- Computes the camera apertures and focal length:
  - `horizontal_aperture = pixel_size * 1e-3 * width`
  - `vertical_aperture = pixel_size * 1e-3 * height`
  - `focal_length_x = fx * pixel_size * 1e-3`
  - `focal_length_y = fy * pixel_size * 1e-3`
  - `focal_length = (focal_length_x + focal_length_y) / 2`
- Sets optical parameters:
  - `set_focal_length(focal_length / 10.0)`
  - `set_focus_distance(focus_distance)`
  - `set_lens_aperture(f_number * 100.0)`
  - `set_horizontal_aperture(horizontal_aperture / 10.0)`
  - `set_vertical_aperture(vertical_aperture / 10.0)`
  - `set_clipping_range(0.05, 1.0e5)`
  - `set_projection_type("pinhole")`
- Recomputes intrinsic matrix `K` on the fly:

  ```python
  fx = width * self.get_focal_length() / self.get_horizontal_aperture()
  fy = height * self.get_focal_length() / self.get_vertical_aperture()
  self.is_camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
  ```

### 4. Outputs from `get_observations()`

`CustomCamera.get_observations()` returns a dict:

- **`color_image`**: RGB image (`H x W x 3`, float32), either from `get_rgba()` or `DiffuseAlbedo` depending on `output_mode`.
- **`depth_image`**: Depth map from `get_depth()` (same resolution as color).
- **`camera2env_pose`**: 4x4 transform from camera to environment, computed from USD prims.
- **`camera_params`**: 3x3 intrinsic matrix `K` as a Python list.

These are the values consumed by tasks (e.g. `banana.py`) for perception and planning.

---

## Summary checklist for a new camera

To add or tweak a camera in a task YAML:

1. **Choose a `name`** and, optionally, a `parent` prim under the task root.
2. **Set pose**: `translation`, `orientation` (quaternion `[w, x, y, z]`), and `camera_axes`.
3. Under `params`, provide:
   - `pixel_size`, `f_number`, `focus_distance`
   - `camera_params = [fx, fy, cx, cy]`
   - `resolution_width`, `resolution_height`
   - optional `output_mode` (`"rgb"` or `"diffuse_albedo"`).
4. Ensure your task (e.g. `banana.py`) constructs `CustomCamera` with this `cfg` (this is already wired up in the current code).
