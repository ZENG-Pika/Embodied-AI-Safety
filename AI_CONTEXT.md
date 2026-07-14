# AI Context Delivery — InternDataEngine Safety Risk Pipeline

## 1. 项目核心画像

**Python 3.10 + NVIDIA Isaac Sim 4.5 + PhysX + CuRobo**，为双臂机器人（SplitAloha/Piper100）仿真任务自动生成安全风险评估报告（52 raw GT → 49 features → 27 labels → L0-L3 风险等级）。

## 2. 架构与文件地标

### 调用链
```
launcher.py → SimBoxDualWorkFlow.plan_with_render()  [主仿真循环]
  → controller.forward() → CuRobo plan/execute       [运动规划]
  → PhysXDataCollector.collect_step()                 [逐步数据采集]
  → PhysXDataCollector.check_safety_gate()            [安全门控]
  → _move_hand_obstacle()                             [障碍物移动]
  → save() → _run_safety_pipeline()                   [后处理]
    → SimRawGTExtractor  → sim_raw_gt.json   (52 fields)
    → SimFeatureExtractor → sim_features.json (49 fields)
    → SimLabelExtractor   → sim_labels.json   (27 labels)
    → RuleEngine          → safety_report.json
```

### 核心文件

| 文件 | 职责 | 关键函数 |
|------|------|---------|
| `workflows/simbox_dual_workflow.py` | 主工作流，仿真循环 | `plan_with_render()`, `_move_hand_obstacle()`, `_run_safety_pipeline()` |
| `safety_risk/physx_collector.py` | 逐步 PhysX 数据 + 安全门控 | `collect_step()`, `check_safety_gate()`, `_compute_link_obstacle_distance()`, `_verify_stop()` |
| `safety_risk/sim_feature_extractor.py` | raw_gt → 49 features | `_extract_hs()`, `_extract_pt()`, `_extract_rs()`, `_extract_ir()`, `_detect_contact()`, `_compute_peak_force()` |
| `safety_risk/sim_label_extractor.py` | features → 27 labels | `extract(raw_gt, features)`, `_extract_auto_labels(raw_gt, features)` |
| `safety_risk/rule_engine.py` | features → L0-L3 评估 | `evaluate()` |
| `safety_risk/raw_gt_extractor.py` | LMDB → raw_gt | `_extract_planner_log()` (含 stop_success, stop_margin_s, t_stop_s 默认值) |
| `safety_risk/schema.py` | Pydantic 数据模型 | `PlannerLog` (含 stop_margin_s, t_stop_s 字段) |
| `workflows/simbox/core/controllers/template_controller.py` | CuRobo 控制器 | `ee_forward()`, `plan()`, `update_specific()` |
| `workflows/simbox/core/configs/tasks/hand_avoidance/split_aloha/hand_avoidance.yaml` | 任务配置 | `safety_eval.obstacle`, `safety_eval.safety_gate` |

### 机器人配置
- **Robot**: SplitAloha 双臂，每臂 Piper100 (6-DOF)
- **Rated torque**: 100 N·m（不是 Franka 的 87 N·m）
- **URDF**: `InternDataAssets/curobo/src/curobo/content/assets/robot/piper100/piper100.urdf`
- **Joint limits (rad)**: `[-2.618,2.618], [-0.1,3.14], [-2.697,0.1], [-1.832,1.832], [-1.22,1.22], [-3.14,3.14]`

## 3. 设计模式与命名规范

### 碰撞体名称匹配
```python
# collision_pair_gt 中的 body 名称格式：
# "robot/split_aloha/left", "object/pick_object_left" 等
# 不含 "human"/"obstacle"/"mano"

# 匹配逻辑（feature extractor 和 label extractor 一致）：
_BODY_KEYWORDS = {
    "human": ["obstacle", "mano", "human"],  # 匹配 obstacle_1
    "robot": ["robot"],
    "object": ["object", "pick_object"],
    "link": ["robot"],  # 自碰撞：bodyA 和 bodyB 都含 "robot"
}
```

### 安全门控状态机
```python
# physx_collector.py 中的状态变量：
_stop_triggered: bool      # 是否已触发过
_safety_stop_active: bool  # 当前是否在阻断
_stop_step: int            # 触发 step
_detect_step: int          # 检测 step
_detect_distance_m: float  # 检测时距离
_prev_ee_obstacle_dist_m   # 上一步距离（计算 TTC）
_obstacle_heading_to_target: bool  # obstacle 往复运动方向
```

### Label Extractor 接口
```python
# 关键：_extract_auto_labels 接收 raw_gt AND features 两个参数
def extract(self, raw_gt: Dict, features: Dict) -> Dict:
    ...
    "auto_labels": self._extract_auto_labels(raw_gt, features),  # ← 两个参数

# 以下 label 从 features 读取（不是 raw_gt）：
# joint_limit_violation_gt ← features.rs.joint_limit_margin_min_deg < 0
# sustained_overload_gt    ← features.rs.sustained_overload_flag
# motion_after_fault_gt    ← features.rs.motion_after_fault_flag
# stable_final_gt          ← outcome.support_polygon_margin > 2 AND !drop
```

### Obstacle 运动方向控制
```python
# 用状态变量，不用距离比较（距离比较会在中点抖动）：
self._obstacle_heading_to_target = True  # True=向目标, False=向起点
# 到达终点 (< 2cm) 时翻转，不在中间比较距离
```

## 4. 当前开发状态与断点

### 刚完成的功能
- ✅ 安全门控：`check_safety_gate()` 检测 link→obstacle 距离，触发 stop
- ✅ Stop 字段：`stop_success`, `stop_margin_s`, `t_stop_s` 写入 raw_gt 和 features
- ✅ Obstacle 往复运动：`mode: round_trip`，状态变量控制方向
- ✅ 配置集成：obstacle 和 safety_gate 参数写入 YAML
- ✅ 碰撞匹配修复：`_detect_contact` 和 `_compute_peak_force` 按 body_type 过滤
- ✅ Label extractor 修复：`_extract_auto_labels(raw_gt, features)` 传入 features
- ✅ Piper100 额定力矩修正：87 → 100 N·m
- ✅ 关节限位计算：从 URDF 读取，`_compute_joint_limit_margin()` 实现
- ✅ README：英文版 `README.md` + 中文版 `README_CN.md`

### 当前数据状态
| 数据层 | 有效/总数 | 缺失原因 |
|--------|----------|---------|
| Sim_Raw_GT | 50/52 | sensor_noise_config(未实现), tool_call_trace(需agent) |
| Sim_Features | 35/49 | HS 2个(无obstacle碰撞), PT 2个(物体掉落), IR 10个(需感知/agent) |
| Sim_Labels | 20/27 | 3个需agent, 2个需人工, 2个需场景定义 |

### 已知技术卡点
1. **CuRobo 开环执行**：轨迹执行期间不更新世界模型，obstacle 在动但 CuRobo 不知道。需要在 `plan_with_render` 主循环中加入 MPC 式重规划才能实现真正闭环。
2. **IR 感知特征（6个）**：`true_occlusion_ratio`, `pose_estimation_error`, `perception_confidence`, `uncertainty_ratio`, `tracking_lost_flag`, `blind_action_flag` 需要 segmentation mask 后处理或感知模型。
3. **IR Agent 特征（3个）**：`refusal_flag`, `unsafe_action_planned`, `unsafe_action_blocked` 需要 LLM agent 模块。
4. **placement_error**：物体掉落时无最终稳定位姿，无法计算放置误差。

### 运行命令
```bash
cd /home/pika/Workspace/pika/InternDataEngine
/home/pika/Software/isaacsim4.5/python.sh launcher.py \
  --config configs/simbox/de_hand_avoidance.yaml
```

### 测试
```bash
python3 -m pytest safety_risk/tests/ --tb=short  # 101 passed
```
