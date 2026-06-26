<div align="center">

# InternDataEngine

**高保真机器人操作合成数据生成器 + 安全风险评估流水线**

</div>

<div align="center">

[![Paper InternData-A1](https://img.shields.io/badge/Paper-InternData--A1-red.svg)](https://arxiv.org/abs/2511.16651)
[![Paper Nimbus](https://img.shields.io/badge/Paper-Nimbus-red.svg)](https://arxiv.org/abs/2601.21449)
[![Paper InternVLA-M1](https://img.shields.io/badge/Paper-InternVLA--M1-red.svg)](https://arxiv.org/abs/2510.13778)
[![Data InternData-A1](https://img.shields.io/badge/Data-InternData--A1-blue?logo=huggingface)](https://huggingface.co/datasets/InternRobotics/InternData-A1)
[![Data InternData-M1](https://img.shields.io/badge/Data-InternData--M1-blue?logo=huggingface)](https://huggingface.co/datasets/InternRobotics/InternData-M1)
[![Docs](https://img.shields.io/badge/Docs-Online-green.svg)](https://internrobotics.github.io/InternDataEngine-Docs/)

</div>

> 🇬🇧 English version: [README.md](README.md)

## 项目简介

InternDataEngine 是一个基于 **NVIDIA Isaac Sim 4.5** 的具身智能合成数据生成引擎。它将高保真物理仿真、语义任务生成和大规模数据生产统一在一个框架中。

本仓库在原始 InternDataEngine 的基础上，扩展了一套 **安全风险评估流水线**，可在仿真过程中自动评估机器人操作的安全性，输出结构化的风险报告，包含 52 个原始 GT 字段、49 个特征、27 个标签，以及 L0-L3 四级风险评估。

### 核心特性

- **高保真物理仿真**：基于 PhysX 引擎，支持刚体、铰接体、可变形物体
- **安全风险流水线**：自动碰撞检测、近距离监控、保护性停止、风险标注
- **多模态数据**：RGB、深度图、分割掩码、关节状态、接触力
- **双臂支持**：SplitAloha 双臂机器人，每臂 6 自由度（Piper100）
- **避障场景**：动态人手（MANO 模型）向机器人工作空间移动

## 仓库结构

```
InternDataEngine/
├── launcher.py                          # 启动入口
├── configs/
│   └── simbox/
│       ├── de_hand_avoidance.yaml       # 避手任务配置
│       └── de_obstacle_avoidance.yaml   # 避障任务配置
├── workflows/
│   └── simbox/
│       ├── simbox_dual_workflow.py      # 主工作流（仿真循环）
│       └── core/
│           ├── controllers/             # CuRobo 运动规划控制器
│           ├── configs/tasks/           # 任务配置
│           └── robots/                  # 机器人配置 (split_aloha.yaml)
├── safety_risk/                         # 安全风险评估流水线
│   ├── schema.py                        # Pydantic 数据模型
│   ├── raw_gt_extractor.py              # 从 LMDB 提取 Sim_Raw_GT
│   ├── sim_raw_extractor.py             # SimRawEpisode 构建器
│   ├── sim_feature_extractor.py         # 计算 49 个 Sim_Features
│   ├── sim_label_extractor.py           # 生成 27 个 Sim_Labels
│   ├── rule_engine.py                   # HS/PT/RS/IR L0-L3 规则评估
│   ├── physx_collector.py               # 逐步 PhysX 数据采集 + 安全门控
│   ├── report.py                        # JSON 安全报告生成
│   ├── workflow_integration.py          # 工作流适配器
│   └── tests/                           # 101 个单元测试
├── robot_safety_risk_data_contract.xlsx # 数据契约（52/49/27 字段定义）
└── output/                              # 生成的 episode 数据
    └── <任务名>/.../<episode_id>/
        ├── sim_raw_gt.json              # 52 个原始 GT 字段
        ├── sim_features.json            # 49 个计算特征
        ├── sim_labels.json              # 27 个风险标签
        ├── safety_reports/*.json        # 最终风险报告
        ├── lmdb/                        # 原始传感器数据
        └── images.*/                    # 提取的图像目录
```

## 快速开始

### 环境要求

- NVIDIA Isaac Sim 4.5（安装路径示例：`/home/pika/Software/isaacsim4.5/`）
- 支持 CUDA 的 GPU
- Python 3.10+（使用 Isaac Sim 自带的 Python）

### 运行仿真

```bash
cd /home/pika/Workspace/pika/InternDataEngine

# 运行避手任务（请将 isaacsim 路径替换为你自己的安装路径）
/home/pika/Software/isaacsim4.5/python.sh launcher.py \
  --config configs/simbox/de_hand_avoidance.yaml
```

> **注意**：请将 `/home/pika/Software/isaacsim4.5/python.sh` 替换为你实际的 Isaac Sim Python 路径。

### 自定义参数运行

```bash
# 单个 episode（1 种布局）
/home/pika/Software/isaacsim4.5/python.sh launcher.py \
  --config configs/simbox/de_hand_avoidance.yaml \
  --load_stage.layout_random_generator.args.random_num=1

# 多个 episode（6 种布局）
/home/pika/Software/isaacsim4.5/python.sh launcher.py \
  --config configs/simbox/de_hand_avoidance.yaml
```

### 运行测试

```bash
python3 -m pytest safety_risk/tests/ --tb=short
```

预期输出：`101 passed`

## 配置说明

### 任务配置

编辑 `workflows/simbox/core/configs/tasks/hand_avoidance/split_aloha/hand_avoidance.yaml`：

```yaml
# 安全风险评估
safety_eval:
  enabled: true
  output_subdir: safety_reports

  # 障碍物（人手）运动配置
  obstacle:
    enabled: true                   # false = 障碍物不动
    name: obstacle_1                # 场景中的物体名称
    target: [-0.10, -0.40, 0.80]   # 目标位置 [x, y, z]
    speed: 0.0035                   # 每步移动距离（米），0.0035 × 30Hz ≈ 0.1 m/s
    fixed_z: 0.80                   # 固定高度（桌面高度）
    mode: round_trip                # "once" = 到目标停, "round_trip" = 往返运动

  # 安全门控（保护性停止）
  safety_gate:
    enabled: true                   # false = 不做安全检查
    distance_threshold_m: 0.30      # 触发停止的距离阈值（米）
    stop_verify_steps: 5            # 停止后验证步数
```

### 流水线配置

编辑 `configs/simbox/de_hand_avoidance.yaml`：

```yaml
name: hand_avoidance
load_stage:
  scene_loader:
    type: env_loader
    args:
      workflow_type: SimBoxDualWorkFlow
      cfg_path: workflows/simbox/core/configs/tasks/hand_avoidance/split_aloha/hand_avoidance.yaml
      simulator:
        physics_dt: 1/30            # 物理仿真频率 30Hz
        rendering_dt: 1/30          # 渲染频率 30Hz
        headless: true              # true = 无界面, false = 显示 GUI
  layout_random_generator:
    type: env_randomizer
    args:
      random_num: 6                 # 布局变体数量
```

## 数据契约

安全风险流水线每次 episode 生成三个结构化数据文件，定义见 `robot_safety_risk_data_contract.xlsx`：

### Sim_Raw_GT（52 个字段）

仿真过程中采集的原始 GT 数据：

| 信号组 | 字段数 | 说明 |
|--------|--------|------|
| 仿真元信息 | 5 | 场景 ID、随机种子、物理/光照/传感器配置 |
| 机器人状态 | 7 | 关节位置、速度、加速度、力矩、link 位姿、EE 位姿 |
| 物体状态 | 4 | 物体位姿、速度、角速度、物理参数 |
| 环境状态 | 4 | 人体位姿、入侵轨迹、场景网格、障碍物位姿 |
| 精确距离 GT | 6 | 机器人-人、EE-人、物体-人、物体-环境、link-环境、自距离 |
| 碰撞 GT | 6 | 碰撞对、碰撞位置、穿透深度、接触力、冲量、持续时间 |
| 抓取 GT | 3 | 接触力、滑移距离、抓取状态 |
| 结果 GT | 4 | 掉落事件、掉落高度、支撑裕度、损坏状态 |
| 规划日志 | 4 | 规划/执行轨迹、安全门控状态、控制命令 |
| 虚拟传感器 | 6 | RGB、深度、分割掩码、实例 ID、边界框、可见性 |
| HRI 日志 | 3 | 用户指令、不安全指令标志、工具调用追踪 |

### Sim_Features（49 个特征）

从 raw GT 计算的风险评估特征，按风险类别组织：

| 风险类 | 特征数 | 关键字段 |
|--------|--------|----------|
| **HS**（人身安全） | 11 | 机器人-人最小距离、EE-人距离、相对速度、TTC、人接触标志、接触力峰值、停止成功、停止裕度 |
| **PT**（物理任务） | 16 | 物体-环境距离、夹爪力、滑移距离、掉落标志、掉落高度、物体碰撞、损坏标志、放置误差 |
| **RS**（机器人安全） | 10 | link-环境距离、自碰撞距离、机器人环境碰撞、自碰撞、关节限位裕度、力矩比、持续过载 |
| **IR**（指令风险） | 12 | 遮挡比例、位姿估计误差、感知置信度、不安全指令、拒绝标志、底层命令 |

### Sim_Labels（27 个标签）

风险标签和评估结果：

| 类别 | 字段数 | 说明 |
|------|--------|------|
| 自动标签 | 18 | 二值标志：碰撞、掉落、损坏、过载等 |
| 任务标签 | 2 | 任务语义成功、场景真实性（需人工标注） |
| 风险标签 | 5 | 根因列表、HS/PT/RS/IR 风险等级（L0-L3） |
| 人工标签 | 2 | 人工覆写风险等级、标注有效性（需人工审核） |
| 评估结果 | 7 | 综合等级、各类等级、触发规则 |

### Safety Report（安全报告）

最终 JSON 报告，汇总所有风险评估结果：

```json
{
  "episode_id": "BananaBaseTask_2026-06-25_...",
  "risk_levels": {"HS": "L2", "PT": "L2", "RS": "L2", "IR": "L0", "overall": "L2"},
  "triggered_rules": [
    {"rule_id": "HS-L2-HIGH-SPEED-NEAR-HUMAN", "level": "L2", "description": "高速接近人手"},
    {"rule_id": "PT-L2-DROP-NO-DAMAGE", "level": "L2", "description": "物体掉落无损坏"}
  ],
  "root_cause": ["high_speed_near_human", "drop_no_damage", ...],
  "key_labels": {
    "human_contact_flag_gt": false,
    "drop_flag_gt": true,
    "damage_flag_gt": false,
    "robot_env_collision_flag_gt": true,
    "self_collision_flag_gt": false,
    "unsafe_instruction_flag_gt": false
  },
  "data_quality": "A",
  "summary": {"overall_level": "L2", "total_rules_triggered": 6, "has_l3_hard_trigger": false}
}
```

## 风险等级定义

| 等级 | 含义 | 示例 |
|------|------|------|
| **L0** | 安全，无风险 | 正常操作，无事件发生 |
| **L1** | 近失事件 | 接近碰撞但成功避免 |
| **L2** | 中等风险 | 物体掉落（无损坏）、轻微碰撞 |
| **L3** | 严重风险 | 撞到人、物体损坏、不可恢复故障 |

### 风险类别

| 类别 | 全称 | 评估内容 |
|------|------|---------|
| **HS** | 人身安全 (Human Safety) | 到人距离、接触力、接近速度、停止成功 |
| **PT** | 物理任务 (Physical Task) | 物体抓取、滑移、掉落、碰撞、损坏、放置精度 |
| **RS** | 机器人安全 (Robot Safety) | 自碰撞、环境碰撞、关节限位、力矩过载 |
| **IR** | 指令风险 (Instruction Risk) | 指令安全性、感知可靠性、动作规划安全性 |

## 安全门控

安全门控在仿真过程中监控机器人到障碍物的距离，当距离低于阈值时触发保护性停止。

### 工作原理

```
每个仿真步骤：
  1. 执行机器人动作（CuRobo 规划的轨迹）
  2. 移动障碍物（人手向机器人靠近）
  3. 采集 PhysX 数据（力、距离、碰撞）
  4. 检查安全门控：
     - 计算所有 robot link 到障碍物的最小距离
     - 如果距离 < 阈值（默认 0.30m）：
       → 清空 action_dict（保持位置）
       → 清除控制器 cmd_plan（防止重新规划）
       → 记录 stop_success、stop_margin、t_stop
```

### 配置参数

在任务 YAML 的 `safety_eval.safety_gate` 下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | true | 启用/禁用安全门控 |
| `distance_threshold_m` | 0.30 | 触发停止的距离阈值（米） |
| `stop_verify_steps` | 5 | 停止后验证机器人是否停住的步数 |

## 流水线架构

```
launcher.py
  → SimBoxDualWorkFlow
    → plan_with_render()          # 主仿真循环
      → CuRobo MotionGen          # 运动规划
      → PhysXDataCollector        # 逐步数据采集 + 安全门控
      → _move_hand_obstacle()     # 移动人手向机器人靠近
    → save()
      → _run_safety_pipeline()
        → SimRawGTExtractor       # LMDB → sim_raw_gt.json（52 字段）
        → SimFeatureExtractor     # raw_gt → sim_features.json（49 特征）
        → SimLabelExtractor       # features → sim_labels.json（27 标签）
        → RuleEngine              # features → 风险评估（L0-L3）
        → ReportGenerator         # labels → safety_report.json
```

## 输出结构

每次 episode 生成以下输出：

```
output/<任务>/<场景>/<episode_id>/
├── sim_raw_gt.json          # 52 个原始 GT 字段（~25MB）
├── sim_features.json        # 49 个计算特征（~9KB）
├── sim_labels.json          # 27 个风险标签（~7KB）
├── safety_reports/
│   └── *_risk.json          # 最终安全报告（~2KB）
├── meta_info.pkl            # episode 元数据
├── lmdb/                    # 原始传感器数据（RGB、深度、分割）
├── images.rgb.head/         # 头部相机 RGB 图像
├── images.depth.head/       # 头部相机深度图
├── images.seg.head/         # 头部相机分割掩码
├── images.rgb.hand_left/    # 左手相机 RGB
└── images.rgb.hand_right/   # 右手相机 RGB
```

## 机器人配置

本仓库使用 **SplitAloha** 双臂机器人，每臂为 **Piper100**：

| 参数 | 值 |
|------|-----|
| 臂数 | 2 × Piper100（每臂 6 自由度） |
| 额定力矩 | 100 N·m / 关节 |
| 关节限位 | 见 `piper100.urdf` |
| 夹爪 | 平行夹爪，最大开口 0.10m |
| EE 轴 | Z 轴 |

机器人配置：`workflows/simbox/core/configs/robots/split_aloha.yaml`

## 许可证与引用

本仓库代码遵循 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可。

```BibTeX
@article{tian2025interndata,
  title={Interndata-a1: Pioneering high-fidelity synthetic data for pre-training generalist policy},
  author={Tian, Yang and Yang, Yuyin and Xie, Yiman and Cai, Zetao and Shi, Xu and Gao, Ning and Liu, Hangxu and Jiang, Xuekun and Qiu, Zherui and Yuan, Feng and others},
  journal={arXiv preprint arXiv:2511.16651},
  year={2025}
}

@article{he2026nimbus,
  title={Nimbus: A Unified Embodied Synthetic Data Generation Framework},
  author={He, Zeyu and Zhang, Yuchang and Zhou, Yuanzhen and Tao, Miao and Li, Hengjie and Tian, Yang and Zeng, Jia and Wang, Tai and Cai, Wenzhe and Chen, Yilun and others},
  journal={arXiv preprint arXiv:2601.21449},
  year={2026}
}

@article{chen2025internvla,
  title={Internvla-m1: A spatially guided vision-language-action framework for generalist robot policy},
  author={Chen, Xinyi and Chen, Yilun and Fu, Yanwei and Gao, Ning and Jia, Jiaya and Jin, Weiyang and Li, Hao and Mu, Yao and Pang, Jiangmiao and Qiao, Yu and others},
  journal={arXiv preprint arXiv:2510.13778},
  year={2025}
}
```
