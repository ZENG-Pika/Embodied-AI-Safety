# improved_curobo 与 Isaac Sim 5.0 集成说明

本文档说明 `feature/improved_curobo-isaacsim5-integration` 分支带来的改进：

1. 把个人分支的 cuRobo 改进整理成最小补丁，供任意 cuRobo 源码（含 Isaac Sim 5.0
   内置 cuRobo 包）复现；
2. 记录 Isaac Sim 5.0（本机为 pip 版）环境的搭建与运行方法；
3. 记录团队基准在 Isaac Sim 5.0 上的验证结果。

补丁与脚本：

- `deps/curobo/improved_curobo_features.patch`：个人 cuRobo 的功能改进
- `deps/curobo/isaac50_torch27_compat.patch`：Isaac Sim 5.0（torch 2.7）兼容修复
- `scripts/sync_improved_curobo.sh`：把上述补丁应用到 cuRobo 源码目录

## 1. improved_curobo 的具体改进

补丁基于 cuRobo `v0.7.5` 发布源码生成（`git apply --check` 已验证可应用）。
两个成功率相关改进（世界缩放、中段种子）在补丁中默认开启，可用环境变量
`=0` 显式关闭；debug 打印类开关默认关闭。

### 1.1 世界坐标系 Cube 缩放开关（`util/usd_helper.py`）

`get_cube_attrs()` 原先只用 Cube prim 自身的 `xformOp:scale` 计算碰撞体尺寸，
父级（如世界场景、道具父 prim）的缩放会被忽略，导致场景缩放后碰撞盒与实际物体
不匹配。

- 开关：`CUROBO_USE_WORLD_CUBE_SCALE=1` 时改用 `get_prim_world_pose()` 返回的
  累计 Local-to-World 缩放（含所有祖先缩放），并修正符号处理。
- 调试：`DEBUG_CUROBO_CUBE_SCALE=1` 会在 `safety_wall` 相关 prim 上打印
  局部/世界缩放与最终尺寸。
- 默认开启（补丁内默认 `1`）；设 `CUROBO_USE_WORLD_CUBE_SCALE=0` 可恢复
  仅用 Cube 局部缩放的历史行为。

### 1.2 轨迹优化线性种子中段插值（`wrap/reacher/trajopt.py`）

在 trajopt 线性种子（`q_start -> q_goal`）上叠加一个中间关节空间凸起，改变
优化初值形态，用于实验中引导轨迹经过特定中段构型。

- 开关：默认开启（补丁内默认 `1`）；设 `CUROBO_MID_SEED_ENABLE=0` 可关闭
- 参数：
  - `CUROBO_MID_SEED_JOINT=1`：作用关节下标（默认 1）
  - `CUROBO_MID_SEED_AMP=0.25`：凸起幅度（默认 0.25，rad）
  - `CUROBO_MID_SEED_DEBUG=1`：打印启用信息
- 凸起在种子两端为 0、中点最大（`sin(pi * t)` 形状），对种子数量、batch、
  action_horizon 均按广播方式生效。

### 1.3 `lerp` -> `curobo_lerp` 重命名（`curobolib/cpp/helper_math.h`）

helper_math.h 中定义的 `lerp(float/2/3/4)` 与 CUDA 12.8+ 工具头中的 `lerp`
符号冲突，导致 Isaac Sim 5.0 环境的 CUDA 编译失败。重命名为 `curobo_lerp`
只影响编译，不改变任何数值语义；该组函数在 cuRobo 内核中没有被调用。

### 1.4 四元数姿态误差梯度重写（`types/math.py`）

`OrientationError` 不再依赖 `torch.jit` 脚本化的模块级 `geodesic_distance`，
改为类内直接计算，并去掉 `current_quat.detach()`。

- 数学等价：官方实现 `grad = grad_out * quat_res/||err||`，重写后
  `grad = grad_out * (1/r_err) * quat_error`（`quat_error` 未归一化，除以
  `r_err` 后一致）。
- 动机：规避 torch 2.7 / Isaac Sim 5.0 环境下 torch.jit 对该路径的兼容问题；
  减少一次 detach 副本。
- Review 注意：`backward` 现在只返回 `current_quat` 的梯度（输入 0
  `goal_quat` 返回 `None`）。在 cuRobo 中 `goal_quat` 通常是常量，不影响
  训练/优化路径；如需对 `goal_quat` 求导请指出。

## 2. Isaac Sim 5.0 torch 2.7 兼容修复（`util/sample_lib.py`）

torch 2.7 中 `torch.Size([n], device=..., dtype=...)` 不再接受关键字参数
（`TypeError: tuple() takes no keyword arguments`）。改为 `torch.Size([n])`，
device/dtype 由调用方（sample 函数）按需处理。这是 Isaac Sim 5.0 环境下的
必要适配，与功能改进分开放置，便于单独评审。

## 3. 补丁应用方法

```bash
# 对任意 cuRobo 源码目录（结构为 src/curobo/...）：
scripts/sync_improved_curobo.sh /path/to/curobo --check   # 只检查，不改文件
scripts/sync_improved_curobo.sh /path/to/curobo           # 实际应用
```

脚本对每个补丁先 `git apply --check` 再应用；失败会停止并提示 cuRobo 版本
可能不匹配。目标目录必须是 git 工作树或至少可用 `git apply`。

> 注意：仓库 `.gitignore` 含 `curobo` 规则，`deps/curobo/` 下的补丁文件需要
> `git add -f deps/curobo/*.patch` 才能纳入提交，避免误吞整个 cuRobo 源码树。

## 4. Isaac Sim 5.0 环境（本机 pip 版）

> 团队基准机器（zxw）使用
> `/home/zxw/isaacsim-5.0/kit/python/lib/python3.11/site-packages/curobo/`
> 内置 cuRobo；本机 pip 版 Isaac Sim 5.0 不内置 cuRobo，通过
> `PYTHONPATH` 指向 cuRobo 源码目录。

已验证环境：

| 项目 | 值 |
| --- | --- |
| Python | 3.11（conda env `env_isaacsim50`） |
| Isaac Sim | `isaacsim[all,extscache]==5.0.0` -> `5.0.0-rc.45+release.23960.184afb15.gl` |
| torch | 2.7.0+cu126 |
| 关键依赖 | ray、open3d-cpu、yourdfpy、drake（pydrake）、omegaconf、lmdb |
| Isaac 固定版本 | numpy==1.26.0、packaging==23.0、pyparsing==3.0.9、psutil==5.9.8 |
| 运行必需 | `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1`（pydrake 静态 TLS） |

安装要点（已在本机执行）：

```bash
conda create -n env_isaacsim50 python=3.11 -y
conda activate env_isaacsim50
pip install 'isaacsim[all,extscache]==5.0.0'
pip install ray open3d-cpu yourdfpy drake omegaconf lmdb
# 按 Isaac 要求固定版本：
pip install numpy==1.26.0 packaging==23.0 pyparsing==3.0.9 psutil==5.9.8
```

注意：`isaacsim` 相关 wheel 来自 `pypi.nvidia.com` 索引；安装体积约十几 GB，
请预留磁盘空间（本机 `/home` 约剩 29GB，一轮 episode 输出约 0.9GB）。

## 5. 运行方式

cuRobo 源码目录：本机为 `/home/burger/InternDataAssets/InternDataAssets/curobo`
（含已编译的 `curobolib/*.so`，Python 3.11）。应用补丁后运行：

```bash
export ISAACSIM_PYTHON=/home/burger/miniconda3/envs/env_isaacsim50/bin/python
export PYTHONPATH=/path/to/curobo/src          # 指向应用补丁后的 cuRobo
export INTERNDATA_ISAAC5_COMPAT=1
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONNOUSERSITE=1
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1

# 功能默认开启；如需关闭或调参才显式导出：
# export CUROBO_USE_WORLD_CUBE_SCALE=0
# export CUROBO_MID_SEED_ENABLE=0
export CUROBO_MID_SEED_JOINT=1      # 可选，中段种子作用关节（默认 1）
export CUROBO_MID_SEED_AMP=0.25     # 可选，凸起幅度 rad（默认 0.25）

cd <repo>
scripts/isaac50/python.sh launcher.py --config configs/simbox/de_hand_avoidance_isaac50.yaml
```

### 本机路径覆盖

`workflows/simbox/isaac50_curobo_configs/piper100_{left,right}_arm.yml` 中
`urdf_path`/`asset_root_path` 仓库内默认是 zxw 机器（Isaac 内置 cuRobo）的绝对
路径；本机运行时请替换为本机 cuRobo 源码内对应路径，例如：

```text
urdf_path: /home/burger/InternDataAssets/InternDataAssets/curobo/src/curobo/content/assets/robot/piper100/piper100.urdf
asset_root_path: /home/burger/InternDataAssets/InternDataAssets/curobo/src/curobo/content/assets/robot
```

该覆盖是本机专用，不应提交进仓库。

## 6. 测试记录（Isaac Sim 5.0）

测试均在 pip 版 Isaac Sim 5.0（`env_isaacsim50`）+ 个人 cuRobo 源码
（`/tmp/curobo_isaac50`，即用户源码 + `sample_lib.py` torch 2.7 修复）上执行。

| 项目 | 命令/输入 | 结果 |
| --- | --- | --- |
| 环境依赖补齐 | 依次安装 ray、open3d-cpu、yourdfpy、drake，并加 `LD_PRELOAD=...libgomp.so.1` | 前 5 轮 `ModuleNotFoundError`/TLS 问题全部解决 |
| 团队基准（无 MANO 变体） | `hand_avoidance` 任务去掉 MANO 障碍物（`ignore_substring` 含 obstacle/mano） | `Task is successful`，`EXIT=0`；完整 episode：LMDB（611MB）、head/hand_left/hand_right/overview 多路视频、safety_reports、sim_raw_gt/features/labels；日志 `/tmp/isaac50_nomano_run2.log`，输出 `output/hand_avoidance_isaac50_nomano/.../2026-09-01_23_30_09_661820/` |
| 团队基准（带 MANO） | 原 `hand_avoidance` 配置 | 本机缺 MANO 资产：`workflows/simbox/example_assets/task/hand_model/mano_hand.usd` 是指向 `InternDataAssets/assets/mano_urdf/mano.usd` 的断链，运行在 `rigid_object.py:51` 报 `IndexError: list index out of range`（约 35 分钟处） |
| improved_curobo 功能开启 | 同上（无 MANO 变体），另设 `CUROBO_USE_WORLD_CUBE_SCALE=1`、`CUROBO_MID_SEED_ENABLE=1`（JOINT=1、AMP=0.25、DEBUG=1） | `Task is successful`，`EXIT=0`；日志出现 18 次 `[CUROBO_MID_SEED] enabled joint=1 amp=0.25 horizon=28 dof=6`，说明改进代码路径实际执行；完整 episode 903MB；日志 `/tmp/isaac50_features_run1.log`，输出 `output/hand_avoidance_isaac50_features/.../2026-09-01_23_51_27_314947/` |
| 团队基准（带 MANO） | 原 `hand_avoidance` 配置 | 本机缺 MANO 资产：`workflows/simbox/example_assets/task/hand_model/mano_hand.usd` 是指向 `InternDataAssets/assets/mano_urdf/mano.usd` 的断链，运行在 `rigid_object.py:51` 报 `IndexError: list index out of range`（约 35 分钟处） |

补丁本身已验证：`git apply --check` 在 cuRobo `v0.7.5` 干净源码上通过。

## 6.1 4.5 vs 5.0 改进有效性对比（2026-09-02）

目的：确认 improved_curobo 迁移到 Isaac Sim 5.0 后仍然有效、行为没有退化。

单元级（同一份改进源码，两边输出逐位一致）：

| 试验 | 4.5（env_interndata45） | 5.0（env_isaacsim50） |
| --- | --- | --- |
| `get_cube_attrs` 局部缩放（开关=0） | dims `[1.0, 2.0, 4.0]` | dims `[1.0, 2.0, 4.0]` |
| `get_cube_attrs` 世界缩放（开关=1，父级 scale=2） | dims `[2.0, 4.0, 8.0]` | dims `[2.0, 4.0, 8.0]` |
| OrientationError 旋转误差/梯度 | `0.7206` / `[0, 0.1545, 0.5497, -0.8210]` | 完全相同 |
| mid-seed 中段凸起（28 步） | 与 5.0 逐位一致 | 与 4.5 逐位一致 |

端到端（同一无 MANO 的 hand_avoidance 变体，功能开启）：

| 运行 | 代码基线 | 结果 | mid-seed 激活 | 未收敛计划 |
| --- | --- | --- | --- | --- |
| 4.5 | 用户完整改进源码（fix/hand-avoidance-runtime @ e3cfc11 + PYTHONPATH） | `Task is successful`，EXIT=0，episode 988MB | 22 次 | 6 |
| 5.0 | 用户完整改进源码（migration @ 8a0bf5a + PYTHONPATH） | `Task is successful`，EXIT=0，episode 903MB | 18 次 | 4 |
| 5.0 | **最小补丁形式**（v0.7.5 + `deps/curobo/*.patch` + 重新编译 curobolib，即团队交付形式） | `Task is successful`，EXIT=0，episode 913MB | 18 次 | 4 |

结论：改进代码本身与运行环境无关（单元结果逐位一致）；最小补丁形式在
Isaac 5.0 上端到端跑通，效果与用户完整源码一致。4.5/5.0 的 mid-seed 激活
次数与未收敛次数略有差异属预期（PhysX/渲染/随机种子在两端不完全一致），
不影响「改进有效」的结论。日志：`/tmp/isaac45_features_run1.log`、
`/tmp/isaac50_features_run1.log`、`/tmp/isaac50_features_patch_run2.log`。

## 7. 已知限制与 Review 要点

1. **MANO 资产缺失**：完整 `hand_avoidance`（带手部障碍物）需要
   `mano_urdf/mano.usd`。本机没有该文件，请用户提供，或确认评审时可跳过。
2. **补丁基线**：补丁针对 cuRobo `v0.7.5` 生成。Isaac Sim 5.0 内置 cuRobo 的
   确切版本以 zxw 机器为准；如不同，先 `--check`，必要时手工合并冲突 hunk。
3. **功能默认开启**：世界缩放与中段种子在补丁中默认开启（`=1`），设 `=0`
   可回退；未应用补丁的 cuRobo（如团队内置包）不受影响。debug 开关保持默认关。
4. **`types/math.py` 梯度**：`goal_quat` 方向梯度现返回 `None`（见 1.4）。
5. **curobolib 编译**：`helper_math.h` 的 `curobo_lerp` 重命名在重新编译
   curobolib 时才生效；使用已编译 `.so` 的环境不受影响。
6. 本分支未改动团队迁移分支的 5.0/3DGS/WorldComposer 兼容代码，也未提交
   piper100 yml 的本机路径覆盖。
