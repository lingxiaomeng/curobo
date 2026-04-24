# 船载机械臂扩展实现细节（动力学改造重点）

## Goal

在不改 CUDA RNEA 内核的前提下，将船体运动预测作为外部扰动接入规划链路，
并复用现有能耗项实现“抗船体扰动 + 节能”联合优化。

本文件是实现细节文档，重点描述动力学模型相关改动与代码落点。

## 1. 当前代码链路（改造前）

### 1.1 关键调用路径

1. Action 序列进入状态转移：
   - `RobotStateTransition.forward(...)`
2. 由状态转移生成完整状态与扭矩：
   - `RobotStateTransition.compute_augmented_state(...)`
3. 扭矩由逆动力学给出：
   - `Dynamics.compute_inverse_dynamics(joint_state, f_ext=None)`
4. 代价中已有能耗项：
   - `aggregate_energy_regularization(...)`

### 1.2 已有能力（可直接复用）

1. `Dynamics.compute_inverse_dynamics` 已支持 `f_ext` 输入和梯度。
2. `CSpace` 的 `STATE` 代价已支持能耗正则（索引 4）。
3. Rollout/Solver 已有参数分发机制，可加统一更新接口。

## 2. 动力学改造总原则

1. 不修改 RNEA CUDA kernel，不改 `RNEAForwardFunction` 参数顺序。
2. 扰动建模放在 Transition 层（`robot_state_transition.py`）完成。
3. 仅在调用逆动力学时注入 `f_ext`，其余链路保持兼容。
4. 默认关闭船体补偿开关，避免现有任务回归。

## 3. 动力学模型改造细节

## 3.1 逆动力学数学接口（保持不变）

逆动力学目标接口保持为：

$$
\tau = \text{ID}(q, \dot{q}, \ddot{q}, f_{ext})
$$

其中：

1. $$q, \dot{q}, \ddot{q}$$ 来自状态转移序列。
2. $$f_{ext}$$ 是每个 link 的 6D 空间力（前三维力矩，后三维力）。

## 3.2 外部扰动张量契约（新增约束）

建议在 Transition 层统一接受以下形状，再传给 Dynamics：

1. 直接输入外力：
   - `[batch, horizon, num_links, 6]`
2. 预展平形状（可选）：
   - `[batch*horizon, num_links, 6]`
3. 单时刻广播输入（可选）：
   - `[num_links, 6]`

最终传给 `Dynamics.compute_inverse_dynamics` 的形状建议统一为：

- `[batch*horizon, num_links, 6]`

## 3.3 船体预测到外力的映射（新增）

### 输入（建议）

1. 船体位姿：`[batch, horizon, 7]`（位置 + 四元数）
2. 船体速度：`[batch, horizon, 6]`（线速度 + 角速度）
3. 船体加速度：`[batch, horizon, 6]`（线加速度 + 角加速度）

### 映射策略（第一版）

采用“基座等效扰动”映射，不做全链高保真流体动力建模：

1. 基座等效力
   - $$F = s \cdot K_f \odot a_{base}$$
2. 基座等效力矩
   - $$\tau_b = s \cdot K_\tau \odot \alpha_{base}$$

说明：

1. $$s$$ 对应 `ship_wrench_sign`。
2. $$K_f, K_\tau$$ 对应可配置增益。
3. 第一版可先忽略二阶耦合项（如 $$\omega \times (I\omega)$$），后续按精度需求再加。

### 坐标系建议

1. 若预测在世界系，需变换到目标 link 局部系后再写入 `f_ext`。
2. 若暂时不做严格变换，可先约定预测已在基座系。
3. 文档中要明确坐标系约定，防止符号方向错误。

## 3.4 核心插入点（必须改）

文件：`curobo/_src/transition/robot_state_transition.py`

当前核心调用：

```python
joint_torque = self.robot_dynamics.compute_inverse_dynamics(state_seq)
```

改造后：

```python
f_ext = self._build_external_wrench_for_state_seq(state_seq)
joint_torque = self.robot_dynamics.compute_inverse_dynamics(state_seq, f_ext=f_ext)
```

其中 `_build_external_wrench_for_state_seq(...)` 为新增内部方法。

## 3.5 batch/horizon 缓冲管理（必须改）

在 Transition 中新增并维护以下缓存：

1. `self._ship_pose_prediction`
2. `self._ship_vel_prediction`
3. `self._ship_acc_prediction`
4. `self._external_wrench_prediction`
5. `self._f_ext_buffer`（建议预分配，避免循环内反复创建）

在 `update_batch_size(...)` 中同步重建 `self._f_ext_buffer`。

## 3.6 梯度与可微性

1. 若 `f_ext.requires_grad=True`，`Dynamics` 现有实现会分配 `grad_f_ext_buf`。
2. 默认建议 ship 预测作为外部输入，不参与优化（`requires_grad=False`）。
3. 后续若要做“扰动估计联合优化”，可直接利用已有反向通路。

## 4. 分文件实现细节（含动力学关联）

## 4.1 `curobo/_src/transition/robot_state_transition_cfg.py`

新增字段（建议）：

1. `enable_ship_motion_compensation: bool = False`
2. `ship_input_mode: str = "base_kinematics"`
3. `ship_apply_link: str = "base_link"`
4. `ship_force_gain: list[float] = [1.0, 1.0, 1.0]`
5. `ship_torque_gain: list[float] = [1.0, 1.0, 1.0]`
6. `ship_wrench_sign: float = 1.0`

实现要求：

1. 在 `create(...)` 中兼容旧配置（未给字段时自动补默认值）。
2. 在 `__post_init__` 中做长度和值域校验。

## 4.2 `curobo/_src/transition/robot_state_transition.py`

新增公开接口：

1. `set_ship_motion_prediction(pose=None, velocity=None, acceleration=None)`
2. `set_external_wrench_prediction(f_ext)`
3. `clear_ship_motion_prediction()`

新增内部方法：

1. `_validate_ship_prediction_shapes(...)`
2. `_convert_ship_prediction_to_wrench(...)`
3. `_build_external_wrench_for_state_seq(...)`

关键改动：

1. 在 `compute_augmented_state(...)` 动力学分支中传入 `f_ext`。
2. 若 `self.robot_dynamics is None`，保持现有逻辑并直接忽略船体补偿。
3. 若 ship 功能未启用或无预测输入，`f_ext=None`。

## 4.3 `curobo/_src/robot/dynamics/dynamics.py`（建议小改）

原则上可不改；但建议增加两个“可维护性增强”：

1. 增加公开属性：
   - `num_links`
   - `num_dof`
2. 增加更明确的 `f_ext` 形状报错信息：
   - `last dim must be 6`
   - `link dim must match num_links`

这样可减少 Transition 层访问私有成员或出现难定位的 reshape 错误。

## 4.4 `curobo/_src/rollout/rollout_robot.py`

新增：

1. `update_ship_motion_prediction(...)`

行为：

1. 同步更新 `transition_model`。
2. 同步更新 `metrics_transition_model`。
3. 支持 `clear=True`。

## 4.5 `curobo/_src/solver/solver_core.py`

新增：

1. `update_ship_motion_prediction(...)`

行为：

1. 遍历 `metrics_rollout`、`optimizer_rollouts`、`auxiliary_rollout`、
   `additional_metrics_rollouts`。
2. 统一调用 rollout 的 `update_ship_motion_prediction(...)`。

## 4.6 `curobo/_src/solver/solver_mpc.py`

新增对外接口：

1. `update_ship_motion_prediction(...)`（透传到 core）

推荐调用序：

1. 更新 current/goal
2. 更新 ship prediction
3. 执行 `optimize_action_sequence(...)`

## 4.7 `curobo/_src/solver/solver_trajopt.py`

新增对外接口：

1. `update_ship_motion_prediction(...)`（透传到 core）

## 4.8 `curobo/_src/motion/motion_planner.py`

新增透传接口：

1. `update_ship_motion_prediction(...)`（透传到 `trajopt_solver`）

## 5. 与能耗项的耦合细节（重要）

当前能耗项来自 `aggregate_energy_regularization`：

$$
J_E = w_E \cdot (\tau \cdot \dot{q} \cdot dt)^2
$$

对应配置索引：

1. `squared_l2_regularization_weight[4]` 即能耗权重。

建议：

1. MPC 初始可从 `0.0 -> 100.0` 开始试。
2. TrajOpt 已有较大能耗权重，可先保留，再按工况微调。

## 6. 配置改造细节

## 6.1 机器人配置（必须）

在 ship 专用 robot yml 中设置：

```yaml
robot_cfg:
  load_dynamics: true
```

否则 Transition 中不会创建 `robot_dynamics`，`f_ext` 注入链路无效。

## 6.2 Transition 配置（新增）

```yaml
transition_model_cfg:
  enable_ship_motion_compensation: true
  ship_input_mode: base_kinematics
  ship_apply_link: base_link
  ship_force_gain: [1.0, 1.0, 1.0]
  ship_torque_gain: [1.0, 1.0, 1.0]
  ship_wrench_sign: 1.0
```

## 6.3 代价配置（启用节能）

```yaml
rollout:
  cost_cfg:
    cspace_cfg:
      cost_type: STATE
      squared_l2_regularization_weight: [v_w, a_w, j_w, tau_w, energy_w]
```

## 7. 运行时接入流程（实现视角）

## 7.1 MPC 闭环

每控制周期：

1. 更新当前关节状态。
2. 更新船体预测（下一个 horizon）。
3. 调用 `optimize_action_sequence` 或 `optimize_next_action`。

## 7.2 TrajOpt / MotionPlanner

每次重规划：

1. 写入当前 ship 预测。
2. 调用 `plan_pose` 或 `solve_pose`。
3. 若预测失效，调用 `clear_ship_motion_prediction` 回退。

## 8. 测试实现细节

## 8.1 动力学层测试（优先级 P0）

文件：`curobo/tests/_src/robot/dynamics/test_rnea_cuda.py`

新增建议：

1. `f_ext=None` 与 `f_ext=0` 结果一致。
2. 单 link 非零 `f_ext` 导致扭矩可观测变化。
3. `f_ext.requires_grad=True` 时梯度非空。

## 8.2 Transition 层测试（优先级 P0）

文件：`curobo/tests/_src/transition/test_robot_state_transition.py`

新增建议：

1. 设置 ship 预测后，`compute_augmented_state` 的 `joint_torque` 发生变化。
2. 修改 batch size 后缓存 shape 仍正确。
3. clear 后行为回到基线。

## 8.3 Solver/Planner 接口测试（优先级 P1）

文件：

1. `curobo/tests/_src/solver/test_solver_mpc.py`
2. `curobo/tests/_src/solver/test_solver_trajopt.py`
3. `curobo/tests/_src/motion/test_motion_planner.py`

新增建议：

1. 新接口存在并可调用。
2. 调用后不破坏原优化流程。
3. 传入非法 shape 时抛出可读错误。

## 9. 性能与稳定性建议

1. `f_ext` 预分配并复用，避免每步新建张量。
2. ship 输入校验只在 shape 变化时做重校验。
3. 数值稳定性优先：增益从小到大调，先验证可行率再压能耗。
4. 默认开关关闭，ship 专用配置显式开启。

## 10. 实施顺序（动力学优先）

1. 先改 `robot_state_transition_cfg.py` 与 `robot_state_transition.py`。
2. 再加 rollout/core/mpc/trajopt/planner 透传接口。
3. 再补 ship 专用配置与示例。
4. 最后完善测试和参数整定。

按上述顺序，可最快得到一个不改内核、可回滚、可验证的船载动力学补偿版本。
