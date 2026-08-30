# WorldComposer NuRec 场景复用流程

本流程适用于每个新场景都提供 `3DGS.usdz` 和 `MESH.zip` 的情况。它保留原始 SimBox 任务 YAML，不替换机器人、桌子、任务物体、控制器或物理参数。每个场景只新增一个独立的 WorldComposer bundle 和一个独立运行配置。

`3DGS.usdz` 是原生 NuRec 视觉层。MESH 是同坐标的几何辅助层：标定和检查时可见，默认隐藏且不参与物理碰撞。两个层同时直接渲染会深度竞争，导致发灰、网格线、闪烁或遮挡，因此不能把 MESH 当普通可见背景使用。

## 0. 输入与前提

每个场景准备两个文件：

- `3DGS.usdz`：必须是 WorldComposer 导出的 NuRec USDZ，不是普通 USDZ 模型。
- `MESH.zip`：应包含 WorldComposer 转换后的 `.usd`、`.usda` 或 `.usdc` 网格。压缩包可包含其他文件，但必须至少有一个可引用的 USD 网格。

本仓库的脚本入口：

- `scripts/worldcomposer/prepare_nurec_bundle.py`
- `scripts/worldcomposer/create_task_calibration_stage.py`
- `scripts/worldcomposer/export_calibration.py`
- `scripts/worldcomposer/create_task_overlay_config.py`
- `scripts/isaac50/open_stage_gui.py`

先确保不存在其他 Isaac Sim 进程。原生 NuRec、RTX 和安全相机占用显存较高，6 GB GPU 不能与 CuRobo SDF 世界或额外模型推理稳定并行。

## 1. 建立场景 bundle

在仓库根目录运行。`my_scene_001` 是新场景 ID，必须唯一。

```bash
python3 scripts/worldcomposer/prepare_nurec_bundle.py \
  --scene-id my_scene_001 \
  --nurec /absolute/path/to/3DGS.usdz \
  --mesh-zip /absolute/path/to/MESH.zip
```

输出目录为：

```text
assets/worldcomposer/my_scene_001/
  3DGS.usdz
  mesh/                         # 原样解包，不重命名或篡改 MESH 依赖
  my_scene_001_Fused.usda       # 相对引用、Z-up、MESH 默认隐藏
  bundle_manifest.json
```

该脚本校验 ZIP 路径安全性、记录 SHA-256，并选取 ZIP 中首选 `.usd` 网格作为引用。若 ZIP 只有 `.ply`，脚本会以退出码 2 停止，不会伪造 USD 网格。

## 2. 只有两份 PLY 时

如果输入是 `scene.ply` 和 `sceneMesh.ply`，先按《场景生成及WorldComposer操作.docx》的前半段执行外部 WorldComposer 流程：

1. 用 `plyfile` 检查类型：3DGS PLY 应有 `f_dc_*`、`opacity`、`scale_*`、`rot_*`；mesh PLY 应有 `face`。
2. 用 `trimesh.load(..., process=False)` 将 mesh PLY 转为 GLB。
3. 在 WorldComposer 中运行 `scene_assembler.py`，得到 NuRec `.usdz` 与 `_mesh.usd`。
4. 将 `.usdz` 和包含 `_mesh.usd` 的 ZIP 作为第 1 步输入。

不要使用 WorldComposer 自动生成的 `_Orin.usd` 作为最终入口；它可能引入额外的 90 度旋转。这里生成的 `*_Fused.usda` 保持 Z-up 和相对引用。

## 3. 以任务资产为基准标定背景

若你的目标是让背景与后续任务的机器人、桌子和目标物一致，应先创建静态任务资产标定场景，而不是先标定空背景：

```bash
python3 scripts/worldcomposer/create_task_calibration_stage.py \
  --bundle-dir assets/worldcomposer/my_scene_001
```

这个命令读取原始 `single_pick/omniobject3d-dish` 任务 YAML、其 arena 和 Franka robot 配置，引用真实任务桌子、Franka、餐盘，并加入与安全场景相同的 MANO 手。对于其他任务，它会引用该 YAML 中全部可直接解析的 arena USD fixtures、机器人和 `objects` USD，而不是只取第一个目标物。它不运行任务，不修改原始 YAML，不启用随机化、控制器、碰撞或物理。生成：

```text
assets/worldcomposer/my_scene_001/my_scene_001_TaskCalibration.usda
```

对其他任务传入该任务 YAML：

```bash
python3 scripts/worldcomposer/create_task_calibration_stage.py \
  --bundle-dir assets/worldcomposer/my_scene_001 \
  --task workflows/simbox/core/configs/tasks/<category>/<task>.yaml
```

例如导入托盘上架任务的全部直接 USD 资产：

```bash
python3 scripts/worldcomposer/create_task_calibration_stage.py \
  --bundle-dir assets/worldcomposer/my_scene_001 \
  --task workflows/simbox/core/configs/tasks/basic/split_aloha/sort_the_tray_on_rack/sort_the_tray_on_rack.yaml
```

脚本会在同名 `.json` 中输出 `fixtures`、`robots`、`objects` 和 `skipped_assets`。`skipped_assets` 非空表示该任务某个资产没有直接 USD 路径，或依赖特定任务类在运行时创建；这类资产不能从 YAML 静态伪造，必须使用相应任务的专用加载流程。

默认名义布局使用原任务桌面：Franka `x=0,y=-0.47`、目标物 `x=0,y=0`。这是用于视觉标定的固定参考，不会覆盖任务运行时的随机采样。需要调整预览参考位姿时使用 `--robot-xy`、`--object-xy`、`--hand-xy`；这些参数只影响静态标定 USD。

## 4. 在 Isaac Sim 中手动标定

打开融合场景，不执行任务：

```bash
scripts/isaac50/python.sh scripts/isaac50/open_stage_gui.py \
  assets/worldcomposer/my_scene_001/my_scene_001_TaskCalibration.usda \
  --nurec \
  --perspective \
  --eye 0 0 0.2 \
  --target 6 0 0.2 \
  --focal-length-mm 18 \
  --width 1280 \
  --height 720
```

对于全景或少视点重建，先将相机保持在 3DGS 有效视域内部；从外部观察到白色云团、黑屏或高斯外壳不代表资产无效。依次尝试朝向 `+X/-X/+Y/-Y`，再调整 `eye`、`target` 与焦距。

保持 `/World/task_0`、`table`、Franka、餐盘和手不动。只选择 `/World/BackgroundCalibration` 并调整其 Translate、Rotate XYZ、Scale。不要移动其子节点 `gauss` 或 `mesh`，也不要对二者分别做不同变换。需要检查 MESH 时，在 Stage 树将 `/World/BackgroundCalibration/mesh` 的 Visibility 临时设为 `inherited`；检查后恢复 `invisible`。

保存到 bundle 内新文件，例如：

```text
assets/worldcomposer/my_scene_001/my_scene_001_calibrated.usda
```

## 5. 导出标定结果

关闭 GUI 后运行：

```bash
scripts/isaac50/python.sh scripts/worldcomposer/export_calibration.py \
  --stage assets/worldcomposer/my_scene_001/my_scene_001_calibrated.usda \
  --output assets/worldcomposer/my_scene_001/calibration.json
```

结果中的 `translation`、`euler_xyz_deg` 和 `scale` 是需要带入任务背景的变换。MESH 必须复用同一组数值。导出文件还会记录静态场景的 `/World/task_0` 平移；它用于让运行配置中的任务资产保持和标定时相同的参考坐标，不能手工套用旧场景的任务偏移。

## 6. 创建任务背景配置

从现有原生 NuRec 配置复制生成一个新的配置，不修改模板或原任务 YAML：

```bash
python3 scripts/worldcomposer/create_task_overlay_config.py \
  --bundle-manifest assets/worldcomposer/my_scene_001/bundle_manifest.json \
  --calibration assets/worldcomposer/my_scene_001/calibration.json \
  --output configs/simbox/worldcomposer_my_scene_001_native.yaml
```

生成配置仅更新：

- `worldcomposer_background`：新 NuRec USDZ 路径与标定变换；
- `worldcomposer_mesh`：新 MESH 路径与相同变换，默认 `visible: false`、`disable_collision: true`；
- `worldcomposer_task_alignment`：从静态标定场景导出的 `/World/task_0` 平移；
- runtime 输出目录和 Kit portable cache 目录。

默认不绑定 NuRec proxy。这是保守设置，避免未验证 MESH 深度代理造成黑屏或遮挡。只有确认 MESH 和 NuRec 像素级配准、且已在独立场景中验证 registered compositing 后，才使用 `--enable-nurec-proxy`。

## 7. 生成任务副本并检查

该命令不执行任务，也不会写回原始任务：

```bash
python3 scripts/run_safety_scenarios.py \
  --config configs/simbox/worldcomposer_my_scene_001_native.yaml \
  --prepare \
  --scenario pick_and_place/franka/single_pick/omniobject3d-dish
```

检查 `.generated/worldcomposer_my_scene_001_native/` 中生成任务的 `worldcomposer_background` 和 `worldcomposer_mesh` 路径及变换。此时不要修改原始 `workflows/simbox/core/configs/tasks/...` 文件。

## 8. 接入任务时的限制

背景和任务资产的相对关系由第 3 步静态场景确定：固定 `/World/task_0`，只移动 `/World/BackgroundCalibration`。第 6 步会将该静态场景的任务根平移写入 `worldcomposer_task_alignment`，所以运行任务不会重用模板中某个旧场景的偏移。若你有意在标定时改变整个任务根，必须同时在静态 USD 中改变 `/World/task_0`，再重新导出；不要只改运行 YAML。

原生 NuRec 在当前 6 GB GPU 上可单独打开，也可在部分轻量模型场景中加载；与 CuRobo 的 MESH SDF 世界、多个 RTX render product、随机干扰物同时运行会发生显存不足。该硬件限制不能通过伪造标定或隐藏错误解决。

## 9. 新场景检查清单

1. `bundle_manifest.json` 状态为 `ready_for_calibration`。
2. `*_Fused.usda` 可由 `open_stage_gui.py --nurec` 打开。
3. 3DGS 与 MESH 使用同一个 `BackgroundCalibration` 变换。
4. MESH 默认隐藏，未开启物理碰撞。
5. `calibration.json` 已保存，且包含与静态场景一致的 `task_root_translation`。
6. 使用新生成的 overlay YAML，未修改原始任务 YAML。
7. 任务执行前只运行 `--prepare` 检查生成副本和资产路径。
