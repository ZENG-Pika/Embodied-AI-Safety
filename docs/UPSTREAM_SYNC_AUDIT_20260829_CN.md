# 上游同步与 Isaac Sim 5.0 差异审计

## 审计基线

- 审计日期：2026-08-29
- 上游包：`Embodied-AI-Safety-fix-hand-avoidance-runtime(3).zip`
- SHA-256：`ad695192f72db6fef4139315f17dd821cd6fdd715e03c58cbfbd7285415904de`
- 上游目录：ZIP 内的 `Embodied-AI-Safety-fix-hand-avoidance-runtime/`
- 本地目录：本仓库根目录

该文档以该 ZIP 为唯一上游代码基线。资产、模型权重、运行输出、3DGS/WorldComposer 场景文件和本机 Isaac Sim 安装不参与逐字节代码同步。

## 审计结论

1. 上游 ZIP 中的 Python 文件在本地均存在。
2. 以 AST 比较类方法和模块级函数后，未发现任何“上游存在、本地缺失”的 Python 符号。
3. 上游 `safety_risk` 的核心实现和配置已直接同步；规则、原始 GT、特征、标签、drop metric、工作流 adapter 与上游一致。
4. 上游手部避障运行时错误分类和重抛逻辑已直接同步至 `base_randomizer.py` 与 `plan_with_render.py`。
5. 本地仅保留 Isaac Sim 5.0 兼容层、3DGS/WorldComposer 背景链路和本审计文档/工具；无关的 LLM 指令分类已删除。后文逐项列出。

“功能完整”在这里指：上游可调用接口和上游安全评估链路均已保留。它不表示 Isaac Sim 4.5 与 5.0 的 PhysX/渲染数值结果逐帧相同；两版的 PhysX 与 Kit 运行时并不保证该性质。

## 已同步的上游功能

| 范围 | 本地位置 | 状态 |
| --- | --- | --- |
| 运行时无效 simulation view 检测与失败分类 | `nimbus/components/load/base_randomizer.py` | 与上游一致 |
| 规划/渲染阶段对致命仿真错误的重抛 | `nimbus_extension/components/plan_with_render/plan_with_render.py` | 与上游一致 |
| Sim raw GT、特征、标签和规则报告 | `safety_risk/` | 核心 Python 模块与上游一致 |
| 风险阈值、信号 schema、任务映射 | `safety_risk/configs/` | 已同步 |
| drop displacement 阈值和证据字段 | `safety_risk/drop_metrics.py`、`workflows/simbox_dual_workflow.py` | 已接入运行时输出 |
| 上游安全评估测试和审计文件 | `safety_risk/tests/`、`safety_risk/audits/` | 已同步 |
| 物体 reset 的 reload/reuse 接口 | `workflows/simbox/core/tasks/banana.py` | 已恢复上游 `_should_reload_object` 与 `_reset_object_velocity` 接口 |

## 保留的 Isaac Sim 5.0 必需差异

| 位置 | 5.0 差异 | 必要原因 | 对上游功能的处理 |
| --- | --- | --- | --- |
| `scripts/isaac50/python.sh`、`scripts/isaac50/run_hand_avoidance.sh` | 用兼容启动包装器启动 Isaac Sim 5.0 | 本机系统的 GLIBC/GLIBCXX 版本不足，不能直接运行原始 `isaac-sim.sh` | 不改变任务配置语义，只替换启动环境和库路径 |
| `configs/simbox/*_isaac50.yaml` | 设置 5.0 portable root、5.0 运行选项 | 隔离 Kit 缓存并避免 Metrics Assembler 的启动崩溃 | 原始 4.5 配置仍保留；5.0 配置为单独入口 |
| `workflows/simbox/core/configs/robots/split_aloha.yaml` | CuRobo robot config 改指向 `isaac50_curobo_configs/` | 5.0 安装不包含上游相对路径中的 4.5 CuRobo 文件 | 内容保持同一 Piper 左右臂运动学定义，只修复可解析路径 |
| `workflows/simbox/core/tasks/banana.py` | 5.0 下不删除和重建已注册刚体；改为 reuse 并归零速度 | 删除 prim 会使 PhysX tensor/contact view 失效，随后出现 `Simulation view object is invalidated` | 非 5.0 分支仍执行上游的 randomization/reload；5.0 保留相同 reset 接口、布局随机化和速度复位，但复用 prim |
| `workflows/simbox/core/tasks/banana.py`、`workflows/simbox_dual_workflow.py` | 在首次 PhysX reset 前修复 MANO 刚体层级、屏蔽异常 UV primvar、使用 5.0 contact view 生命周期 | 5.0 对动态 mesh 碰撞和混合 RigidBodyAPI 层级更严格 | 仅修正 5.0 无法加载或崩溃的 USD 结构；任务对象、碰撞语义和随机布局不改 |
| `workflows/simbox/core/controllers/template_controller.py`、`workflows/simbox/core/robots/split_aloha.py` | 使用 5.0 articulation/XForm 查询、关节索引与初始化顺序 | 4.5 API 和 5.0 articulation handle 生命周期不同 | 保留上游控制器、技能序列和 CuRobo 规划输入输出 |
| `workflows/simbox/core/loggers/*.py` | 适配 5.0 图像/LMDB 写入与异步渲染行为 | 避免 5.0 传感器帧为空或 writer 生命周期异常 | 输出格式仍为上游 LMDB/episode 结构 |
| `core/utils/dr.py`、`region_sampler.py`、`visual_distractor.py` | 适配 5.0 pose/prim 查询和随机化对象更新 | 避免 5.0 下局部/世界坐标与已释放 prim handle 不一致 | 保留随机范围、seed 行为和配置字段 |

## 保留的 3DGS 与审计扩展

这些内容不替换上游的默认控制、任务或安全规则。WorldComposer 仅由专用 3DGS 配置启用；未启用时上游任务布局不变。

| 位置 | 内容 | 默认影响 |
| --- | --- | --- |
| `scripts/worldcomposer/`、`assets/worldcomposer/`、`scenes/` | 3DGS/NuRec 与 WorldComposer 标定、预览、深度 proxy 和背景资产流程 | 不在上游默认配置中启用 |
| `BananaBaseTask` 的 `worldcomposer_*` 配置项 | 将任务资产与已标定的背景变换对齐 | 未配置时平移为 `[0, 0, 0]`，不改变上游布局 |
| `scripts/run_safety_scenarios.py` | 上游 runner 中的 5.0 启动与 3DGS 背景注入分支 | 未指定 `worldcomposer_*` 时保持上游场景内容 |
| `scripts/audit_upstream_python_parity.py` | ZIP 同步的静态结构审计工具 | 不参与仿真运行；用于复核上游文件和符号是否缺失 |

## 严格收敛复核（2026-08-29）

根据“仅保留 5.0 与 3DGS 直接依赖”的要求，本次额外完成以下收敛：

1. 删除本地 `safety_risk/instruction_safety.py`、其单元测试，以及工作流中 LLM API 调用和 hand-avoidance 的 LLM 配置。该模块不属于上游 ZIP，也不参与 Isaac Sim 5.0 或 3DGS 渲染。
2. 将 `configs/simbox/safety_scenarios.yaml` 恢复为上游内容：手部 spawn 区间恢复为 `[0.30, 0.30]` 到 `[0.35, 0.35]`，删除关闭状态的 WorldComposer 示例。
3. 将 `scripts/run_safety_scenarios.py` 中刚体/关节体 `reload_each_episode` 的配置逻辑恢复为上游内容。该逻辑不是 5.0 或 3DGS 的额外要求。
4. 保留 `safety_risk/`、`scripts/run_safety_scenarios.py` 和 `workflows/simbox_dual_workflow.py` 的上游安全与策略功能，因为这些文件本身存在于上游 ZIP，不是本地扩展。

## 可复现审计步骤

从仓库根目录执行。该命令把 ZIP 解压到临时目录，不会修改工作树。

```bash
UPSTREAM_DIR="$(mktemp -d)/upstream"
unzip -q 'Embodied-AI-Safety-fix-hand-avoidance-runtime(3).zip' -d "$UPSTREAM_DIR"
UPSTREAM_DIR="$UPSTREAM_DIR/Embodied-AI-Safety-fix-hand-avoidance-runtime"

# 列出普通源码/配置中的文件差异；本地 5.0、3DGS、模型和输出目录会被排除。
diff -qr "$UPSTREAM_DIR" . \
  -x .git -x output -x failure_output -x assets -x scenes -x models \
  -x '*.usd' -x '*.usda' -x '*.usdz' -x '*.ply' -x '*.mp4' -x '*.zip'

# 检查上游 Python 文件或公开函数是否在本地缺失。必须使用 5.0 的 Python：
# 上游 openpi 子模块含有比系统 Python 更新的语法。
scripts/isaac50/python.sh scripts/audit_upstream_python_parity.py \
  --upstream-root "$UPSTREAM_DIR"
```

审计结果应为零个 `missing_files`、零个 `missing_symbol_files` 和零条 `parse_errors`。在发布或再次同步前，必须重新检查上游 ZIP 的 SHA-256，避免误以不同压缩包为同一上游版本。

## 已完成验证

| 验证 | 结果 |
| --- | --- |
| Python 编译检查 | 已通过：修改后的 `banana.py`、工作流和安全模块均可编译 |
| 安全评估单元测试 | 已通过：`scripts/isaac50/python.sh -m pytest safety_risk/tests -q`，`115 passed`；本地非上游 LLM 测试已删除 |
| 上游文件存在性 | 已通过：`nimbus`、`nimbus_extension`、`configs`、`scripts`、`safety_risk`、`workflows` 的上游文件缺失数均为 0 |
| 严格配置比对 | 已通过：`configs/simbox/safety_scenarios.yaml` 与 hand-avoidance task YAML 均与 ZIP 完全相同 |
| 本地 LLM 路径 | 已删除：源码、配置与运行时引用检索结果均为 0 |
| Isaac Sim 5.0 实际 episode | 已产生成功任务 episode 和原始 GT/特征/标签/风险报告；示例输出位于 `output/hand_avoidance_isaac50_upstream_sync_20260828_12/.../2026-08-28_14_00_18_692767/` |
| 安全与任务判定分离 | 示例 episode 的任务语义 success 已记录，同时风险报告为 L3，证明 risk report 不会覆盖原始 task success |
| 本次二次运行尝试 | 当前受控执行环境未向 Isaac Sim 子进程暴露 NVIDIA 驱动，启动日志为 `NVML_ERROR_DRIVER_NOT_LOADED`、`no CUDA-capable device`；在进入场景加载前已终止。因此它不构成任务成功或失败证据。 |

近期的可视化单 episode 有一次在 pick 后出现 `Plan did not converge` 并停止。这是该随机布局的 CuRobo place 规划失败，不是 Python 接口缺失或 5.0 启动兼容错误；失败 episode 不会生成 success writer 输出。它应作为基准任务的随机规划失败率单独统计，而不应被描述为“上游功能未同步”。

## 维护规则

1. 后续同步以新 ZIP 的 SHA-256 和 branch/commit 为基线，先做文件与 AST 审计，再合并。
2. 不得把 `INTERNDATA_ISAAC5_COMPAT` 的 prim reuse 逻辑删除或改回 4.5 的 delete/reload 路径，否则会重新引入失效 simulation view。
3. 上游新增功能应优先原样合入；若必须改变行为，需在本表新增一行，写明触发错误、修改范围、回退条件和验证证据。
4. 3DGS/WorldComposer、模型推理和 LLM 指令评估属于可选层，不得隐式改变未启用这些配置的上游任务。

## Git 迁移包说明

用于 GitHub 的 Isaac Sim 5.0 迁移分支只提交源码、5.0 配置、WorldComposer
脚本和本文档；不会提交 `assets/` 下的 3DGS/MESH 二进制、`output/`、模型
权重、LMDB、视频或 Kit 缓存。这些文件体积大、属于场景或实验输入，并且已由
`.gitignore` 排除。

复现者需自行提供 WorldComposer 导出的 `3DGS.usdz`（以及需要深度代理时的
`MESH.zip`），然后按照 `docs/worldcomposer_*_workflow_zh.md` 运行 bundle、
标定和 overlay 脚本。这样保留可复现的代码和变换记录，同时不把机器相关或
受数据许可约束的二进制资产混入上游仓库。

在 Ubuntu 20.04 上，`scripts/isaac50/python.sh` 通过仓库内的
`run_in_ubuntu22.sh` 使用 Bubblewrap 运行 Isaac Sim 5.0 所需的 Ubuntu 22.04
rootfs。默认路径为 `/home/zxw/isaacsim-5.0` 和
`/home/zxw/isaacsim-5.0-rootfs`，复现者应通过 `ISAAC_SIM_50_ROOT` 和
`ISAAC_SIM_50_ROOTFS` 覆盖为自己的路径；若宿主系统原生满足 Isaac Sim 5.0
运行要求，可设置 `ISAACSIM_PYTHON=/path/to/isaacsim/python.sh` 跳过容器包装。
