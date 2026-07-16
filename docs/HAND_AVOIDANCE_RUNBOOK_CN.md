# 避手仿真无法启动：原因、修复与正确运行方法

本文记录 `configs/simbox/de_hand_avoidance.yaml` 在更新仓库后首次运行失败的问题、根因、修复方式，以及后续推荐的启动和验收流程。

## 1. 问题概览

最初直接执行：

```bash
cd /home/wp/Embodied-AI-Safety
/home/wp/isaacsim-4.1.0/python.sh launcher.py \
  --config configs/simbox/de_hand_avoidance.yaml
```

仿真无法成功生成 episode。故障分为三个连续阶段：

1. Isaac Sim Python 找不到 `torch`，程序在启动前退出。
2. 补充 PyTorch/CUDA 环境后，MANO 手模型的 USD prim 层级与任务配置不匹配，场景构建失败。
3. 场景失败后残留了已注册的 task；加载器持续重试，原始错误被大量 `Task name should be unique in the world` 错误掩盖。

## 2. 为什么无法运行

### 2.1 Isaac Sim Python 找不到 PyTorch

直接运行时最先出现：

```text
ModuleNotFoundError: No module named 'torch'
```

调用链在导入 `nimbus.utils.random` 时需要 `torch`，但 `/home/wp/isaacsim-4.1.0/python.sh` 默认环境没有包含项目使用的 PyTorch 目录：

```text
/home/wp/isaacsim-4.1.0/torch-cu128
```

因此程序还没有启动 Isaac Sim 场景就已经退出。

### 2.2 MANO USD 默认 prim 与配置预期不一致

补充环境变量后，Isaac Sim 能够启动，但场景构建出现：

```text
RuntimeError: Accessed invalid null prim
```

相关调用位于：

```text
BananaBaseTask.set_up_scene()
  -> BananaBaseTask._load_obj()
  -> RigidObject.__init__()
  -> get_prim_at_path(rigid_prim_path)
```

原配置为：

```yaml
name: obstacle_1
path: task/hand_model/mano_hand.usd
target_class: RigidObject
prim_path_child: mano
```

`mano_hand.usd` 的默认 prim 本身就是 `/mano`。当它被引用到
`/World/task_0/obstacle_1` 时，`mano` 的内容直接组合到对象根节点下，实际可见的是：

```text
/World/task_0/obstacle_1/palm
/World/task_0/obstacle_1/index1y
...
```

而 `RigidObject` 根据 `prim_path_child: mano` 查找：

```text
/World/task_0/obstacle_1/mano
```

该 prim 不存在，所以访问其子节点时触发 `Accessed invalid null prim`。

日志中还会看到类似警告：

```text
Unresolved reference prim path ... mano_physics.usd@</visuals/index1y>
```

这是源 MANO 资产中部分手指视觉子层缺失或版本不一致造成的。它仍会产生警告，但在修复对象根层级后不会阻止本任务运行。

### 2.3 初始化失败后没有清理已注册 task

工作流执行顺序是：

```text
World.add_task(task)
  -> World.reset()
  -> task.set_up_scene()
```

`World.add_task()` 已经注册 task 名称；之后 MANO 对象构建失败，task 仍然留在 `World._current_tasks` 中。

加载器再次尝试相同任务时，又调用：

```python
self.world.add_task(self.task)
```

于是出现：

```text
Exception: Task name should be unique in the world
```

上层数据引擎会继续请求场景，导致该错误高速重复并产生大量日志。这个错误是场景第一次失败后的次生问题，不是最初根因。

## 3. 如何解决

### 3.1 使用正确的 PyTorch/CUDA 环境

启动前为 Isaac Sim Python 补充：

- `PYTHONPATH`：PyTorch CUDA 12.8 环境；
- `LD_LIBRARY_PATH`：cuDNN、NCCL、cuSPARSELt 和 CUDA 动态库；
- `CUDA_HOME`、`PATH` 和 `TORCH_CUDA_ARCH_LIST`。

完整命令见第 4 节。

### 3.2 用 wrapper 保留 `/obstacle_1/mano` 层级

新增/使用：

```text
workflows/simbox/example_assets/task/hand_model/mano_hand_fixed.usda
```

其内容通过额外的 `Root` 包装层引用原 MANO prim：

```usda
#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Root"
{
    def Xform "mano" (
        references = @mano_hand.usd@</mano>
    )
    {
    }
}
```

任务引用 wrapper 后，组合结果包含：

```text
/World/task_0/obstacle_1/mano
/World/task_0/obstacle_1/mano/palm
...
```

这与 `prim_path_child: mano` 一致。

以下两个配置均已改为：

```yaml
path: task/hand_model/mano_hand_fixed.usda
```

- `workflows/simbox/core/configs/tasks/hand_avoidance/split_aloha/hand_avoidance.yaml`
- `workflows/simbox/core/configs/tasks/hand_avoidance/split_aloha/hand_collision_test.yaml`

wrapper 使用相对引用 `@mano_hand.usd@`，不要改回某台机器专用的绝对路径。

### 3.3 初始化失败时清理 World

`nimbus_extension/components/load/env_loader.py` 新增了 `_initialize_workflow_task()`。

如果 `workflow.init_task()` 抛出异常，它会：

1. 将 loader 的 `scene` 置空；
2. 调用 `workflow.world.clear()`；
3. 清除半初始化对象和已注册 task；
4. 重新抛出原始异常。

这样后续重试不会再用 `Task name should be unique in the world` 掩盖真正的场景错误。

## 4. 之后应该如何正确运行

### 4.1 推荐启动命令

最新版仓库已提供便捷脚本。该脚本会自动寻找 Isaac Sim，并配置
PyTorch、CUDA、`LD_LIBRARY_PATH` 和 `TORCH_CUDA_ARCH_LIST`：

```bash
cd /home/wp/Embodied-AI-Safety

ISAACSIM_ROOT=/home/wp/isaacsim-4.1.0 \
./scripts/run_hand_avoidance.sh
```

以下原始命令仍然有效，可用于排查环境变量问题：

```bash
cd /home/wp/Embodied-AI-Safety

export PYTHONPATH=/home/wp/isaacsim-4.1.0/torch-cu128
export LD_LIBRARY_PATH=/home/wp/isaacsim-4.1.0/torch-cu128/nvidia/cudnn/lib:/home/wp/isaacsim-4.1.0/torch-cu128/nvidia/nccl/lib:/home/wp/isaacsim-4.1.0/torch-cu128/nvidia/cusparselt/lib:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=12.0
export PATH=/usr/local/cuda-12.8/bin:${PATH}

/home/wp/isaacsim-4.1.0/python.sh launcher.py \
  --config configs/simbox/de_hand_avoidance.yaml
```

### 4.2 推荐的日志运行方式

使用便捷脚本保存日志：

```bash
cd /home/wp/Embodied-AI-Safety

ISAACSIM_ROOT=/home/wp/isaacsim-4.1.0 \
LOG_FILE=/tmp/hand-avoidance-run.log \
./scripts/run_hand_avoidance.sh
```

也可以使用原始命令重定向日志：

```bash
cd /home/wp/Embodied-AI-Safety

export PYTHONPATH=/home/wp/isaacsim-4.1.0/torch-cu128
export LD_LIBRARY_PATH=/home/wp/isaacsim-4.1.0/torch-cu128/nvidia/cudnn/lib:/home/wp/isaacsim-4.1.0/torch-cu128/nvidia/nccl/lib:/home/wp/isaacsim-4.1.0/torch-cu128/nvidia/cusparselt/lib:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=12.0
export PATH=/usr/local/cuda-12.8/bin:${PATH}

/home/wp/isaacsim-4.1.0/python.sh launcher.py \
  --config configs/simbox/de_hand_avoidance.yaml \
  > /tmp/hand-avoidance-run.log 2>&1
```

在另一个终端查看：

```bash
tail -f /tmp/hand-avoidance-run.log
```

### 4.3 可视化与无界面运行

显示 Isaac Sim 窗口：

```yaml
headless: false
```

无界面批量运行：

```yaml
headless: true
```

该设置位于：

```text
configs/simbox/de_hand_avoidance.yaml
```

### 4.4 启动前快速检查

确认 PyTorch 可导入：

```bash
PYTHONPATH=/home/wp/isaacsim-4.1.0/torch-cu128 \
  /home/wp/isaacsim-4.1.0/python.sh -c \
  "import torch; print(torch.__version__, torch.cuda.is_available())"
```

确认 MANO 文件、wrapper 和本机兼容路径存在：

```bash
test -f workflows/simbox/example_assets/task/hand_model/mano_hand.usd
test -f workflows/simbox/example_assets/task/hand_model/mano_hand_fixed.usda
test -f InternDataAssets/assets/mano_urdf/mano.usd
test -f /home/pika/Workspace/pika/InternDataEngine/InternDataAssets/assets/mano_urdf/mano/mano.usd
```

确认两个任务配置正在使用 wrapper：

```bash
rg -n "mano_hand_fixed.usda" \
  workflows/simbox/core/configs/tasks/hand_avoidance
```

## 5. 如何判断运行成功

成功运行应满足：

1. 日志出现 `Simulation App Startup Complete`；
2. 出现 `pick plan success`、`place plan success`；
3. 出现 `Task is successful`；
4. 进程正常退出，退出码为 0；
5. 新 episode 目录包含以下文件：

```text
sim_raw_gt.json
sim_features.json
sim_labels.json
safety_reports/*_risk.json
```

查找最新原始 GT：

```bash
find output/hand_avoidance -name sim_raw_gt.json \
  -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' \
  | sort | tail
```

验证 JSON 文件没有损坏：

```bash
jq empty /path/to/episode/sim_raw_gt.json
jq empty /path/to/episode/sim_features.json
jq empty /path/to/episode/sim_labels.json
jq empty /path/to/episode/safety_reports/*_risk.json
```

检查 raw GT 的基本信息：

```bash
jq '{episode_id:.episode_meta.episode_id,
     num_steps:.metadata.num_steps,
     warnings:.warnings}' \
  /path/to/episode/sim_raw_gt.json
```

本次修复验证生成的 episode 为：

```text
output/hand_avoidance/BananaBaseTask/split_aloha/hand_avoidance/hand_avoidance/
  2026-07-15_17_44_48_089811/
```

验证结果：

```text
num_steps: 327
warnings: []
data_quality: A
overall risk: L3
```

## 6. 常见日志的含义

### 可以暂时视为非阻塞的警告

以下警告在本次成功运行中仍然出现，但没有阻止 episode 和安全报告生成：

```text
Unresolved reference prim path ... /visuals/*
Duplicate link name ... in articulation metatype
Object already in warp cache
Plan did not converge to a solution
```

其中 `Plan did not converge to a solution` 表示某一次候选规划失败；只要之后出现 `pick plan success` 或 `place plan success`，工作流仍可能正常完成。

### 必须处理的错误

```text
ModuleNotFoundError: No module named 'torch'
RuntimeError: Accessed invalid null prim
Task name should be unique in the world
pipeline failed
```

遇到这些错误时，应从日志中寻找最早出现的 Python 异常。`Task name should be unique in the world` 往往是前一次场景初始化失败后的次生错误。

## 7. 仓库更新注意事项

当前仓库中的 MANO USD 同时依赖仓库根目录相对路径和资产内部的作者绝对路径。
本机已创建以下两个兼容软链接：

```text
/home/wp/Embodied-AI-Safety/InternDataAssets -> /home/wp/InternDataAssets
/home/pika/Workspace/pika/InternDataEngine/InternDataAssets -> /home/wp/InternDataAssets
```

普通 `git pull`、切换分支或更新 tracked 文件不会删除这两个链接，因此通常不需要
重复创建。以下情况需要重新检查或创建：

1. 重新克隆仓库；
2. 换电脑或重装系统；
3. 移动 `/home/wp/InternDataAssets`；
4. 执行会删除未跟踪文件的 `git clean -fd`；
5. 上游再次修改 MANO 资产路径。

重新创建仓库根目录兼容链接：

```bash
cd /home/wp/Embodied-AI-Safety
ln -s /home/wp/InternDataAssets InternDataAssets
```

重新创建作者绝对路径兼容链接（需要管理员权限）：

```bash
sudo mkdir -p /home/pika/Workspace/pika/InternDataEngine
sudo ln -s /home/wp/InternDataAssets \
  /home/pika/Workspace/pika/InternDataEngine/InternDataAssets
```

验证链接：

```bash
readlink -f workflows/simbox/example_assets/task/hand_model/mano_hand.usd
readlink -f /home/pika/Workspace/pika/InternDataEngine/InternDataAssets
```

如果上游后续已经修复为完全相对路径，则 `/home/pika/...` 兼容链接可以删除。

## 8. 自己从 GitHub 更新和上传分支

### 8.1 查看当前状态

进入仓库并确认当前分支、未提交文件和远程地址：

```bash
cd /home/wp/Embodied-AI-Safety

git status --short --branch
git branch -vv
git remote -v
```

本机的 `InternDataAssets` 是兼容软链接，不应提交到 GitHub。`output/` 是运行产物，
也不应作为代码提交。

### 8.2 没有本地修改时更新当前分支

```bash
cd /home/wp/Embodied-AI-Safety

git fetch origin --prune
git pull --ff-only
```

`--ff-only` 可以避免 Git 在不知情的情况下自动生成合并提交。如果提示无法快进，
应先检查本地和远程提交，不要直接使用强制覆盖命令。

### 8.3 有本地修改时更新当前分支

先临时保存 tracked 和 untracked 修改：

```bash
cd /home/wp/Embodied-AI-Safety

git status --short --branch
git stash push -u -m "local changes before pull"
git pull --ff-only
git stash pop
```

如果 `git stash pop` 报冲突，应停止并逐个检查冲突文件，不要使用
`git reset --hard`。确认本机兼容链接仍然存在：

```bash
test -L InternDataAssets
readlink -f InternDataAssets
```

对于已经完成且应该上传的代码，优先先提交到自己的分支，再更新远程；不要长期依赖 stash。

### 8.4 创建自己的开发分支

建议从最新目标分支创建新分支。以下以当前修复分支为基础：

```bash
cd /home/wp/Embodied-AI-Safety

git fetch origin --prune
git switch fix/hand-avoidance-runtime
git pull --ff-only
git switch -c fix/my-change
```

分支名示例：

```text
fix/object-physical-params
fix/mano-asset-path
docs/update-runbook
```

### 8.5 暂存和检查准备上传的文件

只添加本次确实需要提交的文件，不建议直接使用 `git add .`：

```bash
git add path/to/file1 path/to/file2

git status --short
git diff --cached --check
git diff --cached --stat
git diff --cached
```

重点确认暂存区中没有以下内容：

```text
InternDataAssets
output/
/home/wp/... 等本机绝对路径
密码、密钥和访问令牌
```

### 8.6 提交并上传新分支

首次使用时可设置当前仓库的 Git 身份：

```bash
git config user.name "syh-i"
git config user.email "1932957180@qq.com"
```

提交并推送：

```bash
git commit -m "Describe the change"
git push -u origin fix/my-change
```

后续继续向同一分支上传：

```bash
git push
```

上传后可在 GitHub 上从 `fix/my-change` 向目标分支创建 Pull Request。

### 8.7 更新已经存在的远程分支

如果当前分支已经跟踪同名远程分支：

```bash
git status --short --branch
git pull --ff-only

# 修改并检查文件后：
git add path/to/changed-file
git diff --cached --check
git commit -m "Describe the change"
git push
```

不要使用 `git push --force`，除非明确理解它会改写远程提交历史，并已获得其他协作者同意。
