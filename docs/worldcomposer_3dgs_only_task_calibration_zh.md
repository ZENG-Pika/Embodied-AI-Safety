# 纯 3DGS 任务资产标定流程

本流程只使用 WorldComposer 导出的 `3DGS.usdz`。不导入 MESH，不建立 NuRec depth proxy，不向原始任务 YAML 写入任何修改。适用于先显示真实任务资产，再以这些资产为固定参照人工调整 3DGS 背景的场景。

最终会生成一个独立运行 YAML 和一个 `calibration.json`。后者的 `background_parameters_7d` 是给其他人复现的七个参数：`x`、`y`、`z`、`roll_x_deg`、`pitch_y_deg`、`yaw_z_deg`、`scale`。

## 0. 约束

- 纯 3DGS 是视觉背景，不能提供碰撞、物理支撑或精确遮挡深度。
- 标定场景中固定 `/World/task_0`；只变换 `/World/BackgroundCalibration`。
- 七参数模型只有一个 `scale`，因此 Isaac Sim 中 `Scale X/Y/Z` 必须始终相同。若三轴不同，导出脚本会拒绝写入复现参数。
- MESH 相关旧 bundle、配置和脚本不会被本流程修改。

## 1. 创建纯 3DGS bundle

在仓库根目录执行。`SCENE_ID` 必须唯一，下面以当前目录的 `3DGS1.usdz` 为例：

```bash
cd /home/zxw/InternDataEngine/3rd/Embodied-AI-Safety

SCENE_ID=3dgs1_pure_20260828

python3 scripts/worldcomposer/prepare_3dgs_only_bundle.py \
  --scene-id "$SCENE_ID" \
  --nurec "$PWD/3DGS1.usdz"
```

成功时最后一行是：

```text
[worldcomposer] STATUS=SUCCESS step=prepare_3dgs_only_bundle ...
```

输出目录：

```text
assets/worldcomposer/3dgs1_pure_20260828/
  3DGS.usdz
  3dgs1_pure_20260828_3DGS.usda
  bundle_manifest.json
```

该目录不应出现 `mesh/`、`worldcomposer_mesh` 或 proxy 文件。

## 2. 导入指定任务的真实资产，但不执行任务

默认导入 Franka 单物体餐盘任务。对其它任务通过 `--task` 指定其任务 YAML。脚本会导入该任务内能直接引用的全部 fixtures、机器人和 objects，并写入同名 JSON 清单。

例如，导入 Split Aloha 托盘上架任务：

```bash
python3 scripts/worldcomposer/create_task_calibration_stage.py \
  --bundle-dir "assets/worldcomposer/$SCENE_ID" \
  --task workflows/simbox/core/configs/tasks/basic/split_aloha/sort_the_tray_on_rack/sort_the_tray_on_rack.yaml
```

生成：

```text
assets/worldcomposer/3dgs1_pure_20260828/
  3dgs1_pure_20260828_TaskCalibration.usda
  3dgs1_pure_20260828_TaskCalibration.json
```

检查 JSON 的 `fixtures`、`robots`、`objects`，并确认 `skipped_assets` 为 `[]`。若某项在 `skipped_assets` 中，表示任务 YAML 没有可直接引用的 USD，或资产未安装；不能靠静态标定脚本伪造它。

## 3. 在 Isaac Sim 中打开和标定

以下命令只打开场景，不运行控制器、规划器、物理或 policy：

```bash
scripts/isaac50/python.sh scripts/isaac50/open_stage_gui.py \
  "assets/worldcomposer/$SCENE_ID/${SCENE_ID}_TaskCalibration.usda" \
  --nurec \
  --perspective \
  --eye 0 0 0.2 \
  --target 6 0 0.2 \
  --focal-length-mm 18 \
  --width 1280 \
  --height 720
```

在 Stage 面板确认下列结构：

```text
/World/BackgroundCalibration/gauss
/World/task_0/table
/World/task_0/<robot>
/World/task_0/<task objects>
```

操作规则：

1. 保持 `/World/task_0` 及其所有子节点不动。
2. 选择 `/World/BackgroundCalibration`，只调整 `Translate`、`Rotate XYZ`、`Scale`。
3. 先用 Global 坐标系做整体平移和旋转；需要沿背景自身轴微调时才用 Local。
4. 每次 Scale 修改都输入同一个数值到 X、Y、Z，例如 `0.82, 0.82, 0.82`。
5. 不要旋转 `/World` 或 `/World/task_0`，否则任务桌子、机器人和物体会一起移动。
6. 保存为新文件，例如 `assets/worldcomposer/$SCENE_ID/${SCENE_ID}_calibrated.usda`。

## 4. 导出七个复现参数

关闭 GUI 后执行：

```bash
scripts/isaac50/python.sh scripts/worldcomposer/export_calibration.py \
  --stage "assets/worldcomposer/$SCENE_ID/${SCENE_ID}_calibrated.usda" \
  --output "assets/worldcomposer/$SCENE_ID/calibration.json" \
  --require-uniform-scale
```

在 JSON 中读取：

```json
"background_parameters_7d": {
  "x": 0.0,
  "y": 0.0,
  "z": 0.0,
  "roll_x_deg": 0.0,
  "pitch_y_deg": 0.0,
  "yaw_z_deg": 0.0,
  "scale": 1.0
}
```

这七个参数唯一对应 `/World/BackgroundCalibration` 的世界平移、XYZ 欧拉角和统一缩放。导出完成以以下行为准：

```text
[worldcomposer] STATUS=SUCCESS step=export_calibration ...
```

## 5. 生成不含 MESH 的独立运行配置

这一步不修改模板和原任务 YAML：

```bash
python3 scripts/worldcomposer/create_task_overlay_config.py \
  --bundle-manifest "assets/worldcomposer/$SCENE_ID/bundle_manifest.json" \
  --calibration "assets/worldcomposer/$SCENE_ID/calibration.json" \
  --output "configs/simbox/worldcomposer_${SCENE_ID}_3dgs_only.yaml"
```

生成的 YAML 包含 `worldcomposer_background` 和 `worldcomposer_task_alignment`，不会包含 `worldcomposer_mesh`。检查：

```bash
rg -n 'worldcomposer_(background|mesh|task_alignment)' \
  "configs/simbox/worldcomposer_${SCENE_ID}_3dgs_only.yaml"
```

预期只显示 `worldcomposer_background` 和 `worldcomposer_task_alignment`。运行前可先生成任务副本而不执行 episode：

```bash
python3 scripts/run_safety_scenarios.py \
  --config "configs/simbox/worldcomposer_${SCENE_ID}_3dgs_only.yaml" \
  --prepare \
  --scenario pick_and_place/franka/single_pick/omniobject3d-dish
```

## 6. 交付给他人

交付以下文件即可复现同一背景配置：

```text
assets/worldcomposer/<SCENE_ID>/3DGS.usdz
assets/worldcomposer/<SCENE_ID>/bundle_manifest.json
assets/worldcomposer/<SCENE_ID>/calibration.json
configs/simbox/worldcomposer_<SCENE_ID>_3dgs_only.yaml
```

对方需要相同版本的 Isaac Sim 5、同一任务资产池和本仓库脚本。纯 3DGS 没有 MESH 深度代理，因此不同相机视角下仿真物体与背景之间仍可能发生视觉遮挡不一致；该限制不影响七参数变换的复现。
