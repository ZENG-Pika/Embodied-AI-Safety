# Normalization Statistics

Here we provide instructions for computing **normalization statistics** for both **real-world**, **simulation (InternData-A1)** and **sim2real** tasks. The computed statistics are saved in JSON format and are intended to be reused during training and evaluation in the OpenPI pipeline.

Normalization is computed over:
- `state`
- `actions`

and follows the exact data preprocessing and repacking logic used during training.

---

## 1. Simulation Tasks (InternData-A1)
This script `scripts/compute_norm_stats_sim.py` computes normalization statistics for simulation tasks in the InternData-A1 benchmark.

### Supported Robots
- `split_aloha`
- `lift2`
- `genie1`
- `franka`

### Dataset Structure
Download the InternData-A1 datasets from [here](https://huggingface.co/datasets/InternRobotics/InternData-A1).
The structure of the dataset is as follows:

```
InternData-A1/sim/
└── <task_category>/
    └── <robot_name>/
        └── <task_name>/               # no subtask
            ├── data/
            ├── meta/
            └── videos/
```

Some tasks may have subtasks / collections:

```
InternData-A1/sim/
└── <task_category>/
    └── <robot_name>/
        └── <task_name>/
            └── <collect_name>/
                ├── data/
                ├── meta/
                └── videos/
```

### Usage
```
python scripts/compute_norm_stats_sim.py \
  --root_data_dir InternData-A1/sim \
  --task_category pick_and_place_tasks \
  --save_path stats/sim \
  --start_ratio 0.0 \
  --end_ratio 1.0
```

Arguments
- `root_data_dir`: Root directory of simulation datasets.
- `task_category`: Task category to process (e.g. pick_and_place_tasks).
- `save_path`: Root directory where normalization statistics will be saved.
- `start_ratio`, `end_ratio`: Fraction of tasks to process (useful for sharding large datasets).

### Output Structure
```
<save_path>/
└── <task_category>/
    └── <robot_name>/
        └── <task_name>/
            └── <collect_name>/   # empty if no subtask
                └── norm_stats.json
```
During pretraining, set the `stats_dir` argument in `DataConfig` to the `save_path` here.

## 2. Real-World Tasks
This script `scripts/compute_norm_stats_real.py` computes normalization statistics for real-world tasks.

### Supported Robots
- `lift2`
- `split_aloha`
- `acone`
- `genie1`

### Dataset Structure
Real-world datasets are expected to follow the LeRobot repository structure:
```
InternData-A1/real/
    └── <robot_name>/
        └── <task_name>/
            └── <collect_name>/   # empty if no subtask
                ├── data/
                ├── meta/
                └── videos/
```

Example task path:
```
InternData-A1/real/genie1/
└── Pick_a_bag_of_bread_with_the_left_arm__then_handover/set_0
```

### Usage
```
python scripts/compute_norm_stats_real.py \
  --task_path InternData-A1/real/genie1/Pick_a_bag_of_bread_with_the_left_arm__then_handover/* \
  --robot_name genie1 \
  --save_path stats/real
```

Arguments
- `task_path`: Path (or glob pattern) to a real-world task dataset(e.g. `InternData-A1/real/genie1/Pick_a_bag_of_bread_with_the_left_arm__then_handover/*`)
- `robot_name`: Robot platform name (must be supported).
- `save_path`: Root directory where normalization statistics will be saved.

### Output Structure
```
<save_path>/
└── <robot_name>/
    └── <task_name>/
        └── norm_stats.json
```
During finetuning, set the `fixed_stats_dir` argument in `DataConfig` to `<save_path>/<robot_name>/<task_name>` here.

## 3. Sim2Real Experiments
This script `scripts/compute_norm_stats_sim2real.py` computes normalization statistics for sim2real experiments.

### Supported Robots
- `lift2`

### Dataset Structure
Dataset from InternData-A1 are expected to follow the LeRobot repository structure:
```
InternData-A1/sim/
    └── <task_category>/
        └── <robot_name>/
            └── <task_name>/
                └── <collect_name>/
                    ├── data/
                    ├── meta/
                    └── videos/
```

Example task path:
```
InternData-A1/sim/long_horizon_tasks/lift2/
└── sort_the_rubbish
    └── Sort_rubbish_1l2r
    └── Sort_rubbish_2l1r
    └── Sort_rubbish_2l2r
```

### Usage
```
python scripts/compute_norm_stats_sim2real.py \
  --task_path InternData-A1/sim/long_horizon_tasks/lift2/sort_the_rubbish/* \
  --robot_name lift2 \
  --save_path stats/sim2real
```

Arguments
- `task_path`: Path (or glob pattern) to a task dataset(e.g. `InternData-A1/sim/long_horizon_tasks/lift2/sort_the_rubbish/*` means training on all the collections in the task)
- `robot_name`: Robot platform name (we only support `lift2` for now, but you can try other robots).
- `save_path`: Root directory where normalization statistics will be saved.

### Output Structure
```
<save_path>/
└── <robot_name>/
    └── <task_name>/
        └── norm_stats.json
```
During finetuning, set the `fixed_stats_dir` argument in `DataConfig` to `<save_path>/<robot_name>/<task_name>` here.

## Implementation Notes

For simulation tasks and sim2real experiments, computation may stop early (e.g. after 10k steps) to limit runtime.

For sim2real transfer, we set the gripper dimension in the state vector to zero because the state of the gripper in the real world during inference is not aligned with the state in the simulation. See `src/openpi/policies/sim2real_split_aloha_policy.py` for more details.