<div align="center">

# InternDataEngine

**High-Fidelity Synthetic Data Generator for Robotic Manipulation with Safety Risk Evaluation**

</div>

<div align="center">

[![Paper InternData-A1](https://img.shields.io/badge/Paper-InternData--A1-red.svg)](https://arxiv.org/abs/2511.16651)
[![Paper Nimbus](https://img.shields.io/badge/Paper-Nimbus-red.svg)](https://arxiv.org/abs/2601.21449)
[![Paper InternVLA-M1](https://img.shields.io/badge/Paper-InternVLA--M1-red.svg)](https://arxiv.org/abs/2510.13778)
[![Data InternData-A1](https://img.shields.io/badge/Data-InternData--A1-blue?logo=huggingface)](https://huggingface.co/datasets/InternRobotics/InternData-A1)
[![Data InternData-M1](https://img.shields.io/badge/Data-InternData--M1-blue?logo=huggingface)](https://huggingface.co/datasets/InternRobotics/InternData-M1)
[![Docs](https://img.shields.io/badge/Docs-Online-green.svg)](https://internrobotics.github.io/InternDataEngine-Docs/)

</div>

> 🇨🇳 中文版本请见 [README_CN.md](README_CN.md)

## About

InternDataEngine is a synthetic data generation engine for embodied AI, built on **NVIDIA Isaac Sim 4.5**. It unifies high-fidelity physical simulation, semantic task generation, and large-scale data production.

This repository extends the original InternDataEngine with a **Safety Risk Evaluation Pipeline** that automatically assesses robot manipulation safety during simulation, producing structured risk reports with 52 raw GT fields, 49 features, 27 labels, and 4-level risk evaluation (HS/PT/RS/IR).

### Key Features

- **Realistic physics simulation**: Rigid, articulated, deformable objects with PhysX engine
- **Safety risk pipeline**: Automatic collision detection, proximity monitoring, protective stop, risk labeling
- **Multi-modal data**: RGB, depth, segmentation masks, joint states, contact forces
- **Dual-arm support**: SplitAloha dual-arm robot with Piper100 arms
- **Obstacle avoidance**: Dynamic human hand (MANO model) approaching robot workspace

## Repository Structure

```
InternDataEngine/
├── launcher.py                          # Entry point
├── configs/
│   └── simbox/
│       ├── de_hand_avoidance.yaml       # Hand avoidance task config
│       └── de_obstacle_avoidance.yaml   # Obstacle avoidance task config
├── workflows/
│   └── simbox/
│       ├── simbox_dual_workflow.py      # Main workflow (simulation loop)
│       └── core/
│           ├── controllers/             # CuRobo motion planning controllers
│           ├── configs/tasks/           # Task-specific configs
│           └── robots/                  # Robot configs (split_aloha.yaml)
├── safety_risk/                         # Safety risk evaluation pipeline
│   ├── schema.py                        # Pydantic data models
│   ├── raw_gt_extractor.py              # Extract Sim_Raw_GT from LMDB
│   ├── sim_raw_extractor.py             # SimRawEpisode builder
│   ├── sim_feature_extractor.py         # Compute 49 Sim_Features
│   ├── sim_label_extractor.py           # Generate 27 Sim_Labels
│   ├── rule_engine.py                   # HS/PT/RS/IR L0-L3 rule evaluation
│   ├── physx_collector.py               # Per-step PhysX data + safety gate
│   ├── report.py                        # JSON safety report generation
│   ├── workflow_integration.py          # Workflow adapter
│   └── tests/                           # 101 unit tests
├── robot_safety_risk_data_contract.xlsx # Data contract (52/49/27 field specs)
└── output/                              # Generated episodes
    └── <task_name>/.../<episode_id>/
        ├── sim_raw_gt.json              # 52 raw ground truth fields
        ├── sim_features.json            # 49 computed features
        ├── sim_labels.json              # 27 risk labels
        ├── safety_reports/*.json        # Final risk report
        ├── lmdb/                        # Raw sensor data (RGB, depth, seg)
        └── images.*/                    # Extracted image directories
```

## Quick Start

### Prerequisites

- NVIDIA Isaac Sim 4.5 installed at `/home/pika/Software/isaacsim4.5/` (adjust path below)
- CUDA-capable GPU
- Python 3.10+ (Isaac Sim's bundled Python)

### Run a Simulation

```bash
cd /home/pika/Workspace/pika/InternDataEngine

# Run hand avoidance task (adjust isaacsim path to your installation)
/home/pika/Software/isaacsim4.5/python.sh launcher.py \
  --config configs/simbox/de_hand_avoidance.yaml
```

> **Note**: Replace `/home/pika/Software/isaacsim4.5/python.sh` with your Isaac Sim Python path.

### Run with Custom Parameters

```bash
# Single episode (1 layout variation)
/home/pika/Software/isaacsim4.5/python.sh launcher.py \
  --config configs/simbox/de_hand_avoidance.yaml \
  --load_stage.layout_random_generator.args.random_num=1

# Multiple episodes (6 layout variations)
/home/pika/Software/isaacsim4.5/python.sh launcher.py \
  --config configs/simbox/de_hand_avoidance.yaml
```

### Run Tests

```bash
python3 -m pytest safety_risk/tests/ --tb=short
```

Expected output: `101 passed`

## Configuration

### Task Configuration

Edit `workflows/simbox/core/configs/tasks/hand_avoidance/split_aloha/hand_avoidance.yaml`:

```yaml
# Safety risk evaluation
safety_eval:
  enabled: true
  output_subdir: safety_reports

  # Obstacle (human hand) movement
  obstacle:
    enabled: true                   # false = static obstacle
    name: obstacle_1                # object name in scene
    target: [-0.10, -0.40, 0.80]   # target position [x, y, z]
    speed: 0.0035                   # m/step (0.0035 × 30Hz ≈ 0.1 m/s)
    fixed_z: 0.80                   # fixed height on table
    mode: round_trip                # "once" or "round_trip"

  # Safety gate (protective stop)
  safety_gate:
    enabled: true                   # false = no safety check
    distance_threshold_m: 0.30      # trigger stop when link-obstacle < 0.30m
    stop_verify_steps: 5            # steps to verify robot halted
```

### Pipeline Configuration

Edit `configs/simbox/de_hand_avoidance.yaml`:

```yaml
name: hand_avoidance
load_stage:
  scene_loader:
    type: env_loader
    args:
      workflow_type: SimBoxDualWorkFlow
      cfg_path: workflows/simbox/core/configs/tasks/hand_avoidance/split_aloha/hand_avoidance.yaml
      simulator:
        physics_dt: 1/30            # 30Hz physics
        rendering_dt: 1/30          # 30Hz rendering
        headless: true              # true = no GUI, false = show GUI
  layout_random_generator:
    type: env_randomizer
    args:
      random_num: 6                 # number of layout variations
```

## Data Contract

The safety risk pipeline produces three structured data files per episode, defined in `robot_safety_risk_data_contract.xlsx`:

### Sim_Raw_GT (52 fields)

Raw ground truth data collected during simulation.

| Signal Group | Fields | Description |
|-------------|--------|-------------|
| Episode Meta | 5 | scenario_id, random_seed, physics/lighting/sensor config |
| Robot State | 7 | joint positions, velocities, accelerations, torques, link poses, EE pose |
| Object State | 4 | object poses, velocities, angular velocities, physical params |
| Environment | 4 | human body pose, intrusion trajectory, scene mesh, obstacle pose |
| Distance GT | 6 | robot-human, EE-human, object-human, object-env, link-env, self distance |
| Collision GT | 6 | collision pairs, locations, penetration depth, forces, impulses, duration |
| Gripper GT | 3 | contact force, slip distance, grasp state |
| Outcome GT | 4 | drop event, drop height, support margin, damage state |
| Planner Log | 4 | planned/executed trajectory, safety gate status, commands |
| Sensor GT | 6 | RGB, depth, segmentation, instance ID, bbox, visibility |
| HRI Log | 3 | user command, unsafe instruction flag, tool call trace |

### Sim_Features (49 fields)

Computed features for risk evaluation, organized by risk class:

| Risk Class | Features | Key Fields |
|-----------|----------|------------|
| **HS** (Human Safety) | 11 | d_robot_h_min, d_ee_h_min, v_rel_h, TTC_h, human_contact_flag, F_h_peak, stop_success, stop_margin |
| **PT** (Physical Task) | 16 | d_obj_env_min, gripper_force, slip_distance, drop_flag, h_drop, object_collision, damage_flag, placement_error |
| **RS** (Robot Safety) | 10 | d_link_env_min, d_self_min, robot_env_collision, self_collision, joint_limit_margin, torque_ratio, sustained_overload |
| **IR** (Instruction Risk) | 12 | true_occlusion, pose_estimation_error, perception_confidence, unsafe_instruction, refusal_flag, low_level_command |

### Sim_Labels (27 fields)

Risk labels and evaluation results:

| Category | Fields | Description |
|----------|--------|-------------|
| Auto Labels | 18 | Binary flags: contact, drop, damage, collision, overload, etc. |
| Task Labels | 2 | task_semantic_success, scenario_realism (require human annotation) |
| Risk Labels | 5 | root_cause_auto, HS/PT/RS/IR risk levels (L0-L3) |
| Manual Labels | 2 | risk_label_manual_override, annotation_validity (require human review) |
| Evaluation | 7 | overall_level, per-class levels, triggered rules |

### Safety Report

Final JSON report combining all risk assessments:

```json
{
  "episode_id": "BananaBaseTask_2026-06-25_...",
  "risk_levels": {"HS": "L2", "PT": "L2", "RS": "L2", "IR": "L0", "overall": "L2"},
  "triggered_rules": [
    {"rule_id": "HS-L2-HIGH-SPEED-NEAR-HUMAN", "level": "L2", "description": "..."},
    {"rule_id": "PT-L2-DROP-NO-DAMAGE", "level": "L2", "description": "..."}
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

## Risk Level Definitions

| Level | Meaning | Example |
|-------|---------|---------|
| **L0** | Safe, no risk | Normal operation, no incidents |
| **L1** | Near miss | Close to collision but avoided |
| **L2** | Moderate risk | Object dropped (no damage), minor collision |
| **L3** | Severe risk | Human contact, object damage, unrecoverable fault |

### Risk Classes

| Class | Full Name | What It Evaluates |
|-------|-----------|-------------------|
| **HS** | Human Safety | Distance to human, contact force, approach speed, stop success |
| **PT** | Physical Task | Object grasp, slip, drop, collision, damage, placement accuracy |
| **RS** | Robot Safety | Self-collision, environment collision, joint limits, torque overload |
| **IR** | Instruction Risk | Command safety, perception reliability, action planning safety |

## Safety Gate

The safety gate monitors robot-to-obstacle distance during simulation and triggers a protective stop when the distance falls below a threshold.

### How It Works

```
Each simulation step:
  1. Execute robot action (CuRobo planned trajectory)
  2. Move obstacle (human hand approaches robot)
  3. Collect PhysX data (forces, distances, collisions)
  4. Check safety gate:
     - Compute min distance from all robot links to obstacle
     - If distance < threshold (default 0.30m):
       → Clear action_dict (hold position)
       → Clear controller cmd_plan (prevent replanning)
       → Record stop_success, stop_margin, t_stop
```

### Configuration

In task YAML under `safety_eval.safety_gate`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | true | Enable/disable safety gate |
| `distance_threshold_m` | 0.30 | Distance threshold to trigger stop (meters) |
| `stop_verify_steps` | 5 | Steps after stop to verify robot halted |

## Pipeline Architecture

```
launcher.py
  → SimBoxDualWorkFlow
    → plan_with_render()          # Main simulation loop
      → CuRobo MotionGen          # Motion planning
      → PhysXDataCollector        # Per-step data collection + safety gate
      → _move_hand_obstacle()     # Move human hand toward robot
    → save()
      → _run_safety_pipeline()
        → SimRawGTExtractor       # LMDB → sim_raw_gt.json (52 fields)
        → SimFeatureExtractor     # raw_gt → sim_features.json (49 features)
        → SimLabelExtractor       # features → sim_labels.json (27 labels)
        → RuleEngine              # features → risk evaluation (L0-L3)
        → ReportGenerator         # labels → safety_report.json
```

## Output Structure

Each episode generates the following output:

```
output/<task>/<scene>/<episode_id>/
├── sim_raw_gt.json          # 52 raw ground truth fields (~25MB)
├── sim_features.json        # 49 computed features (~9KB)
├── sim_labels.json          # 27 risk labels (~7KB)
├── safety_reports/
│   └── *_risk.json          # Final safety report (~2KB)
├── meta_info.pkl            # Episode metadata
├── lmdb/                    # Raw sensor data (RGB, depth, segmentation)
├── images.rgb.head/         # RGB images from head camera
├── images.depth.head/       # Depth images from head camera
├── images.seg.head/         # Segmentation masks from head camera
├── images.rgb.hand_left/    # RGB from left hand camera
└── images.rgb.hand_right/   # RGB from right hand camera
```

## Robot Configuration

This repository uses the **SplitAloha** dual-arm robot with **Piper100** arms:

| Parameter | Value |
|-----------|-------|
| Arms | 2 × Piper100 (6-DOF each) |
| Rated torque | 100 N·m per joint |
| Joint limits | See `piper100.urdf` |
| Gripper | Parallel jaw, 0.10m max width |
| EE axis | Z-axis |

Robot config: `workflows/simbox/core/configs/robots/split_aloha.yaml`

## License and Citation

All code within this repo is under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

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
