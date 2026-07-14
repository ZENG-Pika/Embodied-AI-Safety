# 当前问题记录

## 日期：2026-07-02（更新于 2026-07-06）

## 1. MANO 手部模型物理层级问题

### 问题描述

MANO 手部 USD 文件的物理层级配置不正确，导致 PhysX 报错。

### 错误信息

```
[Error] Rigid Body of (/World/task_0/obstacle_1/mano/palm) missing xformstack reset 
when child of rigid body (/World/task_0/obstacle_1/mano) in hierarchy.

[Error] CreateJoint - no bodies defined at body0 and body1, joint prim: 
/World/task_0/obstacle_1/mano/joints/j_index1y
```

### 根本原因

MANO USD 文件的物理层级配置：

```
/World/task_0/obstacle_1/mano
  ├── RigidBodyAPI  ← 父级不该有这个
  ├── MassAPI       ← 父级不该有这个
  │
  ├── palm (RigidBody) ← 子级有
  ├── index1y (RigidBody) ← 子级有
  └── ...
```

**PhysX 规则**：父子不能同时有 `RigidBodyAPI`，除非子级做了 XformStack 重置。

### 解决方案（已实现）

在 `_fix_obstacle_physics_hierarchy()` 中：

1. 移除父级 `mano` 的 `RigidBodyAPI` 和 `MassAPI`
2. 添加 `ArticulationRootAPI` 到父级
3. 修复子级碰撞体近似类型为 `convexHull`

```python
# 移除父级 RigidBodyAPI
prim.RemoveAPI(UsdPhysics.RigidBodyAPI)

# 添加 Articulation Root API
UsdPhysics.ArticulationRootAPI.Apply(prim)

# 修复子级碰撞体
for child in prim.GetChildren():
    if child.HasAPI(UsdPhysics.CollisionAPI):
        approx.Set("convexHull")
```

### 影响

- ~~当前：obstacle 的 contact view 无法正常工作，无法获取机器人与人手的真实接触力~~
- **✅ 已修复**：2026-07-06 仿真验证成功，252 个碰撞事件，力 6.2–331.8N

---

## 2. obstacle 碰撞检测

### ~~问题描述~~ ✅ 已解决

~~`collision_pair_gt` 只记录了机械臂与抓取物体的碰撞，没有 obstacle 碰撞记录。~~

### 修复内容

1. 修复 MANO 物理层级（ArticulationRootAPI 方案）
2. 新增独立的 `obstaclecontact_views`
3. 遍历机器人树下所有 RigidBodyAPI
4. 关闭 30cm 安全门控

### 当前状态

| 碰撞对 | 能否检测 | 说明 |
|--------|---------|------|
| robot → pick_object | ✅ | 正常 |
| robot → obstacle_1 | ✅ | 252 个事件，6.2–331.8N |

### 待优化：细分到 link 级别

当前 `bodyA` 记为 `robot/split_aloha/all`，需要改为具体 link。

contact force matrix 形状：`(MANO_body_count, robot_link_count, 3)`

修复方案：按 robot link 维度（第二维）逐项处理，输出格式：
```json
{
  "bodyA": "robot/split_aloha/fr/link4",
  "bodyB": "obstacle/obstacle_1",
  "step": 269,
  "force_n": 42.7
}
```

同一步接触多个 link 时分别输出多条记录。

---

## 3. 数据格式变更记录

### 已完成的修改

| 字段 | 修改内容 |
|------|---------|
| `link_pose_gt` | 合并 `all_link_pose_gt`，包含所有 32 个 link |
| `link_velocity_gt` | 添加角速度计算 |
| `ee_human_distance_gt` | 改为左右臂分别计算 |
| `object_human_distance_gt` | 改为所有物体分别计算 |
| `object_env_distance_gt` | 改为每个物体到所有其他物体的距离 |
| `self_distance_gt` | 改为所有臂部 link 两两距离 |
| `object_physical_params` | 移除 `hazard_class`、`fragility_class` |
| 所有距离单位 | 统一为米 (m) |

### 待修复

| 问题 | 优先级 | 说明 |
|------|--------|------|
| ~~obstacle 碰撞检测~~ | ~~高~~ | ✅ 已解决 |
| obstacle 碰撞细分到 link | 高 | contact force matrix 按 link 维度逐项处理 |
| mass_kg 获取 | 中 | USD prim 没有 MassAPI |
| friction 获取 | 中 | USD prim 没有 CollisionAPI |
| inertia 获取 | 中 | USD prim 没有设置 |

---

## 4. 运行命令

```bash
cd /home/pika/Workspace/pika/InternDataEngine
/home/pika/Software/isaacsim4.5/python.sh launcher.py \
  --config configs/simbox/de_hand_avoidance.yaml
```

## 5. 测试

```bash
python3 -m pytest safety_risk/tests/ --tb=short
# 101 passed
```
