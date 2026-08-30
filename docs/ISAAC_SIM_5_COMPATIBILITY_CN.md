# Embodied-AI-Safety 的 Isaac Sim 5.0 中文兼容说明

## 1. 文档目的

本文说明本机目录

```text
/home/zxw/InternDataEngine/3rd/Embodied-AI-Safety
```

相对上游 `fix/hand-avoidance-runtime` 分支做了哪些修改、每项修改解决什么
Isaac Sim 4.5 到 5.0 的兼容问题，以及如何运行和检查任务输出。

对比基线为：

```text
/home/zxw/InternDataEngine/3rd/Embodied-AI-Safety/Embodied-AI-Safety-fix-hand-avoidance-runtime.zip
```

该 ZIP 于 2026-08-13 从 GitHub 分支下载并同步。同步时保留了上游新增的
`hand_avoidance` runtime、安全场景配置和 `safety_risk` 模块；本地额外内容仅为本文件
所列的 Isaac Sim 5.0 启动、API、PhysX/Fabric 生命周期和资产路径兼容层。

本次审计遵循以下原则：

1. 保留原仓库的任务定义、CuRobo 规划、抓取判定和放置判定。
2. 只在 Isaac Sim 5.0 行为或 API 与 4.5 不兼容时修改源码。
3. Isaac Sim 5.0 专用逻辑尽量受 `INTERNDATA_ISAAC5_COMPAT=1` 保护，避免改变原 4.5 分支。
4. 不使用物体跟随机械臂、直接写物体轨迹或瞬移到目标位置来伪造抓取。

## 2. 当前运行环境

| 项目 | 当前值 |
| --- | --- |
| Isaac Sim | `5.0.0-rc.45+release.23960.184afb15.gl` |
| Isaac Sim 路径 | `/home/zxw/isaacsim-5.0` |
| 兼容用户态 | Ubuntu 22.04.5 LTS rootfs |
| rootfs 路径 | `/home/zxw/isaacsim-5.0-rootfs` |
| GPU 使用方式 | 透传宿主机 NVIDIA 设备、驱动库和 Vulkan/EGL 配置 |
| InternData-A1 资产 | `/home/zxw/InternDataEngine/InternDataAssets` |
| 已验收任务 | Split ALOHA `hand_avoidance` |

宿主机系统库不满足 Isaac Sim 5.0 所需的 `GLIBC`/`GLIBCXX` 版本，因此不能直接运行
`/home/zxw/isaacsim-5.0/python.sh`。本项目通过 bubblewrap 进入 Ubuntu 22.04
用户态，同时继续使用宿主机 GPU。

## 3. 执行链路

命令的实际调用关系如下：

```text
scripts/isaac50/run_hand_avoidance.sh
  -> scripts/isaac50/python.sh
  -> /home/zxw/InternDataEngine/scripts/isaac50/python.sh
  -> /home/zxw/InternDataEngine/scripts/isaac50/run_in_ubuntu22.sh
  -> bwrap + Ubuntu 22.04 rootfs
  -> /home/zxw/isaacsim-5.0/python.sh launcher.py
  -> configs/simbox/de_hand_avoidance_isaac50.yaml
  -> SimBoxDualWorkFlow
  -> BananaBaseTask + Split ALOHA + CuRobo Pick/Place
  -> LMDB、相机视频和安全风险 JSON
```

`run_in_ubuntu22.sh` 会设置：

```text
INTERNDATA_ISAAC5_COMPAT=1
OMNI_KIT_ACCEPT_EULA=YES
PYTHONNOUSERSITE=1
```

其中 `INTERNDATA_ISAAC5_COMPAT=1` 是所有 Isaac Sim 5.0 专用分支的开关。

## 4. 新增的兼容文件

### 4.1 `configs/simbox/de_hand_avoidance_isaac50.yaml`

这是 Isaac Sim 5.0 专用入口配置。它仍然引用原任务配置：

```text
workflows/simbox/core/configs/tasks/hand_avoidance/split_aloha/hand_avoidance.yaml
```

专用配置增加或调整了：

- `headless: true`：默认无界面运行，可用 CLI 嵌套参数覆盖。
- `portable_root: output/.isaac50_portable`：把 Kit 可写缓存放在项目输出目录。
- `disable_metrics_assembler: true`：避免 Isaac 5 在 PhysX tensor view 建立后重新组合引用资产。
- `strict_mode: false`：随机布局失败时继续尝试后续布局。
- 输出目录改为 `output/hand_avoidance_isaac50/`，与原 4.5 输出分开。

### 4.2 `scripts/isaac50/`

- `run_hand_avoidance.sh`：固定正确配置并支持 `RANDOM_SEED`。
- `python.sh`：转发到 InternDataEngine 顶层的 Isaac 5 Python 包装器。
- 顶层 `scripts/isaac50/run_in_ubuntu22.sh`：创建 bubblewrap 用户态并透传 GPU、显示服务、
  `/home`、`/tmp`、CUDA 11.8 和 NVIDIA 驱动文件。

### 4.3 `workflows/simbox/isaac50_curobo_configs/`

包含：

```text
piper100_left_arm.yml
piper100_right_arm.yml
```

原仓库假定 CuRobo 以源码子模块存在于
`workflows/simbox/curobo/src/curobo/...`。Isaac Sim 5.0 使用内置 Python 包，目录结构不同，
因此在仓库内保存 Split ALOHA 左、右臂所需的机器人配置，并让机器人 YAML 指向这里。

### 4.4 本地资产链接和 MANO 包装 USD

- `InternDataAssets` 链接到共享 InternData-A1 资产目录，替换原作者机器上的绝对路径。
- `workflows/simbox/curobo/src/curobo` 链接到 Isaac 5 已安装的 CuRobo 包。
- 本地 `mano_hand.usd` 包装文件修复 ZIP 中无法恢复的外部符号链接，并保留原手模型层级。

这些修改改变的是资产寻址方式，不改变任务目标或技能顺序。

上游 ZIP 中的 `mano_hand.usd` 是指向其作者本机 InternDataAssets 的符号链接；本机共享资产
目录未包含该链接目标。为使 Isaac Sim 5 能实际加载场景，本地文件保留上游要求的
`Root/mano` 层级，并引用已有的 `human_hand_mano.usda` 可视网格和凸包碰撞网格。这是缺失资产
的本机解析修复，不会给物体施加轨迹、接触力或抓取约束。

## 5. 修改的源码文件

### 5.1 `nimbus_extension/components/load/env_loader.py`

修改内容：

1. 在创建 `SimulationApp` 之前处理 `portable_root`。
2. 按配置关闭 metrics assembler 的 operation/change listener。

原因：Isaac Sim 5 的 Kit 缓存需要可写目录；metrics assembler 在加载 4.x 引用资产时可能在
tensor/contact view 生效后改写 stage，导致 PhysX view 失效。

对任务行为的影响：只影响应用启动和 stage 生命周期，不改变 Pick/Place 逻辑。

### 5.2 `workflows/simbox/core/configs/robots/split_aloha.yaml`

修改内容：把左右臂 `robot_file` 从原 CuRobo 子模块路径改到
`workflows/simbox/isaac50_curobo_configs/`。

原因：Isaac Sim 5 安装包没有原仓库预期的源码子模块路径。此前的
`FileNotFoundError: piper100_left_arm.yml` 和把 YAML 当 URDF 的错误均由该路径不匹配引起。

### 5.3 `workflows/simbox/core/robots/split_aloha.py`

修改内容：

- 从配置读取 `body_indices`。
- 最大关节速度覆盖移动底座、左右臂的实际索引数量，不再固定写死 12。
- 初始关节位置同时写入 `body_home`、双臂 home 和双夹爪 home。

原因：Isaac Sim 5 对 articulation 的未初始化移动底座关节更敏感。若只初始化双臂，
`dummy_base_rotate/x/y` 会出现关节漂移、负质量/惯量相关告警放大和控制不稳定。

对任务行为的影响：让机器人以配置给定的 home 状态稳定启动；不增加新的动作策略。

### 5.4 `workflows/simbox/core/tasks/banana.py`

修改内容：

- Isaac 5 分支在首次 `World.reset()` 前把动态物体和机器人注册到 scene。
- tensor/contact view 建立后不再删除并重建同一路径下的动态物体。
- 保存随机采样后的物体位姿和位姿坐标系，在 `post_reset()` 后原样恢复。
- 对世界坐标采样使用 world pose，对其他采样保留 local pose。
- MANO 手模型只在父、子同时含 RigidBody 时移除父级冲突 API，且在首次 PhysX parse 前完成。
- 只有非 kinematic 刚体才清零线速度和角速度，避免
  `Body must be non-kinematic` 错误。

原因：Isaac Sim 5 的 PhysX tensor/contact view 不允许其匹配的 prim 在运行中被删除重建；
reset 还可能恢复 authored pose，因此需要保存而不是重新随机采样。

随机性说明：资产和布局仍由原随机流选择，保存/恢复不会生成第二个随机结果，也不会把物体
固定到人工指定终点。

### 5.5 `workflows/simbox/core/utils/region_sampler.py`

修改内容：

- `A_on_B_region_sampler` 允许传入缓存后的物体底部偏移和目标 bbox。
- A-on-B 采样统一使用世界坐标和 world orientation。
- 放置高度按 `target_z_max + object_bottom_offset + 0.001` 计算。

原因：原实现混用了 world bbox 和 local pose，在含父级 transform 的 Isaac 5 stage 中会造成
物体悬空、穿透或两个物体重合。修改后仍采用原随机平移和随机 yaw，只修正坐标系。

### 5.6 `workflows/simbox_dual_workflow.py`

修改内容：

1. 在任务和资产构建前设置随机种子，布局采样前再次设置同一 seed。
2. Isaac 5 中复用 World 已注册的 task，避免相同 prim/contact view 重复创建。
3. Isaac 5 单任务 stage 使用资产已有碰撞关系，不调用行为已变化的 inverted collision filter。
4. warmup 时保持机器人所有关节在 home/当前目标位置，防止控制器建立前 articulation 漂移。
5. 首次恢复所有物体采样位姿；随后只保持显式标记 `warmup_hold_pose` 的障碍物。
6. 在预规划前完成 10 帧稳定步骤，避免“规划起点”和“开始执行时的真实关节状态”不一致。
7. MANO hierarchy 修复移到 task 初始化阶段，禁止在 tensor view 建立后再修改 PhysX hierarchy。

原因：这些都是 Isaac 5 stage/tensor 生命周期、碰撞组语义和 articulation drive 时序变化导致的
兼容问题。技能仍按原仓库的双臂 Pick 后 Place 顺序执行。

### 5.7 `workflows/simbox/core/controllers/template_controller.py`

修改内容：

- Isaac 5 分支从 PhysX tensor/Fabric 读取实时末端、基座和普通 prim 位姿，避免读取陈旧 USD
  transform。
- 预规划结果按目标位姿缓存；四元数 `q` 与 `-q` 归一为同一个 key。
- 执行缓存轨迹前检查当前关节起点，最大偏差必须小于 `0.05 rad`，否则重新调用 CuRobo。
- CuRobo action 明确映射到当前机械臂索引，不把 arm-only 轨迹误写到移动底座或夹爪。
- 轨迹末端继续保持最后一个关节目标，直到原技能的收敛判定完成。
- 每次新目标规划前清除旧 `cmd_plan`，防止新技能误继续旧轨迹。

原因：Isaac 5 的实时物理状态主要由 Fabric/PhysX 持有，USD transform 可能落后；双臂顺序执行时
预规划与实际开始状态也更容易产生偏差。

重要边界：该文件没有直接设置被抓物体位姿，没有“物体跟随夹爪”逻辑。物体移动仍由刚体、
碰撞、夹爪接触和原 CuRobo attachment/技能流程产生。

### 5.8 `workflows/simbox/core/loggers/lmdb_logger.py`

修改内容：用 OpenCV `VideoWriter` 替换 `imageio.mimsave`。

原因：Isaac Sim 5 Python 环境没有 imageio 的可选 FFmpeg 插件。新实现仍以 15 FPS 输出 MP4，
并校验帧尺寸、RGB/RGBA 通道和 writer 是否成功打开。

对数据的影响：LMDB 内容和传感器帧不变，只改变 MP4 编码入口。

## 6. 已恢复为原仓库内容的文件

以下文件在 2026-08-13 同步后的工作树中直接来自 `fix/hand-avoidance-runtime` 上游：

```text
workflows/simbox/core/skills/pick.py
workflows/simbox/core/skills/place.py
workflows/simbox/core/configs/tasks/hand_avoidance/split_aloha/hand_avoidance.yaml
workflows/simbox/core/loggers/utils.py
```

这意味着：

- Pick/Place 技能调用顺序和成功条件使用原仓库实现。
- 没有额外的物体跟随、物体轨迹写入或瞬移抓取。
- 没有额外 release settle 延迟改变放置动作。
- 原任务 YAML 的对象、目标、技能和阈值保持不变。

## 7. 已删除的非必要差异

审计时删除了以下曾用于排查问题但不属于 4.5 到 5.0 必要兼容的内容：

- kinematic tracked-object follower：会让物体按夹爪位姿被写动，不符合原物理语义。
- release settle steps：原仓库没有该额外等待参数。
- 未使用的 reverse trajectory/retreat 实验代码。
- 未生效的 planner 参数和摩擦实验字段。
- 大量临时 debug 分支、诊断日志和 logger fallback。

删除后重新运行了完整任务，确认仍能产生成功 episode 和数据输出。

## 8. 运行命令

所有命令都从项目目录执行：

```bash
cd /home/zxw/InternDataEngine/3rd/Embodied-AI-Safety
```

### 8.1 单个 episode，无界面，固定 seed

```bash
RANDOM_SEED=0 scripts/isaac50/run_hand_avoidance.sh \
  --load_stage.layout_random_generator.args.random_num=1
```

### 8.2 单个 episode，可视化

```bash
RANDOM_SEED=0 scripts/isaac50/run_hand_avoidance.sh \
  --load_stage.layout_random_generator.args.random_num=1 \
  --load_stage.scene_loader.args.simulator.headless=false
```

注意：不能写 `--headless false`，因为根配置不存在 `headless`。必须覆盖完整嵌套路径：

```text
--load_stage.scene_loader.args.simulator.headless=false
```

### 8.3 多个随机布局

```bash
RANDOM_SEED=0 scripts/isaac50/run_hand_avoidance.sh \
  --load_stage.layout_random_generator.args.random_num=12
```

`random_num=12` 表示从同一可复现随机流生成 12 个布局，不保证每个布局都成功。原仓库只将通过
任务成功条件的 rollout 写成完整 episode；失败次数会记录在 `de_time_profile_*.log` 中。

### 8.4 不使用项目包装脚本的等价可视化命令

```bash
/home/zxw/InternDataEngine/scripts/isaac50/python.sh launcher.py \
  --config configs/simbox/de_hand_avoidance_isaac50.yaml \
  --random_seed 0 \
  --load_stage.layout_random_generator.args.random_num=1 \
  --load_stage.scene_loader.args.simulator.headless=false
```

不要直接使用 `/home/zxw/isaacsim-5.0/python.sh`，否则会再次遇到宿主机
`GLIBC_2.32/2.33/2.34` 和 `GLIBCXX_3.4.29/3.4.30` 缺失问题。

## 9. 输出位置和检查方法

成功 episode 位于：

```text
output/hand_avoidance_isaac50/BananaBaseTask/split_aloha/hand_avoidance/hand_avoidance/<时间戳>/
```

运行日志位于：

```text
output/hand_avoidance_isaac50/de_time_profile_<时间戳>.log
```

查看最新 episode：

```bash
find output/hand_avoidance_isaac50/BananaBaseTask/split_aloha/hand_avoidance/hand_avoidance \
  -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -1
```

查看最近任务成功率和保存记录：

```bash
rg "success rate|save data" output/hand_avoidance_isaac50/de_time_profile_*.log | tail -20
```

完整成功目录通常包含：

- `lmdb/data.mdb` 和 `lmdb/lock.mdb`：逐帧 episode 数据。
- `sim_raw_gt.json`：统一原始 GT 信号。
- `sim_features.json`：HS/PT/RS/IR 风险特征。
- `sim_labels.json`：规则风险标签和证据。
- 自动安全风险报告 JSON。
- 左手、右手和头部相机 MP4。

## 10. 最终验收结果

### 10.1 同步后检查（2026-08-13）

上游 ZIP 已同步到当前工作树；冲突仅出现在 `BananaBaseTask` 的对象重载与
`SimBoxDualWorkFlow` 的预规划稳定步骤。最终合并保留上游的 policy 和
`reload_each_episode` 行为，并且仅在 `INTERNDATA_ISAAC5_COMPAT=1` 下禁止删除已被
Isaac 5 tensor/contact view 注册的对象，同时在非 policy 的 CuRobo 路径中执行稳定预热。

静态 `py_compile` 已通过。同步后单 episode 运行可完成场景加载、MANO 障碍物加载和 CuRobo
规划，但该随机布局按上游成功条件以 `success rate: 0/1` 结束，未产生新成功数据。随机布局
失败是上游 benchmark 的正常结果，不能把它记录为“同步后的成功验收”。随后一次 3-layout
尝试在 Isaac 启动阶段遭遇临时的宿主 GPU/Vulkan 会话错误
`NVML_ERROR_DRIVER_NOT_LOADED` / `ERROR_INCOMPATIBLE_DRIVER` 并退出；检查确认没有残留
Isaac Sim 进程。

因此，当前同步验收结论是“可加载并进入规划，代码静态有效”；需要在 GPU 会话稳定时再次运行
多布局批次，得到新的成功 episode 后才能称为“新分支端到端成功”。

### 10.2 上一兼容基线的端到端成功结果

删除非兼容实验改动后，2026-08-10 重新运行 Isaac Sim 5.0，得到成功 episode：

```text
output/hand_avoidance_isaac50/BananaBaseTask/split_aloha/hand_avoidance/hand_avoidance/2026-08-10_13_10_25_435451
```

验收数据：

| 项目 | 结果 |
| --- | --- |
| random seed | `1868503102` |
| 仿真步数 | 340 |
| LMDB `data.mdb` | 74,256,384 bytes |
| `sim_raw_gt.json` | 36,754,971 bytes |
| `sim_features.json` | 8,473 bytes |
| `sim_labels.json` | 6,515 bytes |
| 自动风险报告 | 已生成 |
| 相机视频 | 3 个，均可由 OpenCV 解码 |
| 视频规格 | 每路 340 帧，640x480 |

同一批次日志先出现一次失败，之后成功保存该 episode，随后又有一次失败，即停止前为 1/3。
这说明布局和规划结果不是固定成功，也不是固定失败；与原仓库一样，随机任务可能因可达性、碰撞
或技能成功条件失败。

## 11. “与原仓库一样”的准确含义

当前实现保持一致的是：任务配置、技能语义、CuRobo 规划入口、抓取/放置判定和成功数据写入
逻辑。Isaac Sim 4.5 和 5.0 使用不同的 PhysX、Fabric、Kit 和 API 版本，因此不能承诺每一帧
轨迹、接触冲量或随机布局成功率逐位相同。

合理的兼容验收标准是：

1. 同一任务定义和随机种子可启动。
2. CuRobo 真实生成抓取和放置轨迹。
3. 物体由物理接触/原 attachment 流程移动，而不是人为写轨迹。
4. 完成原技能的抓取、搬运、释放和放置成功判定。
5. 能输出 LMDB、三路视频和安全风险数据。

当前 Split ALOHA `hand_avoidance` 已达到上述标准。其他任务配置尚未逐项做同等级的 Isaac 5.0
端到端验收，不能仅凭该任务成功就宣称全部原仓库任务已经兼容。
