# 船载机械臂扩展实施方案（详细版）

## 目标

本方案用于将当前 GPU 机器人规划库扩展到船载机械臂场景，满足以下目标：

1. 在规划与控制中引入船体运动预测影响。
2. 复用现有逆动力学与代价框架，尽量不改底层 CUDA 核。
3. 同时覆盖离线轨迹规划（MotionPlanner 和 TrajOpt）与在线闭环控制（MPC）。
4. 第一版优先复用现有能耗项，快速验证效果。

## 范围与边界

### 本方案包含

1. 船体预测输入接入（位姿、速度、加速度时序）。
2. 通过外力项 f_ext 注入到逆动力学计算链路。
3. 通过配置启用现有能耗项进行节能优化。
4. 配置、示例、测试与验收流程。

### 本方案暂不包含

1. 新 CUDA 动力学核开发。
2. 船体流体动力学高保真模型。
3. 电机效率、液压损耗等复杂专用能耗模型。

## 当前可复用能力

1. 逆动力学已支持外力输入 f_ext 与梯度。
2. CSpace 的 STATE 代价已包含能耗正则项（基于扭矩和关节速度）。
3. Rollout 与 Solver 已有统一参数分发机制。
4. MotionPlanner、TrajOpt、MPC 已有高层包装与示例。

这意味着第一版只需在状态转移层打通扰动注入，并在上层增加统一更新接口，即可完成端到端闭环。

## 输入与数据契约（建议）

为避免接口碎片化，建议统一采用以下输入契约：

1. 船体位姿时序
   - 形状: [batch, horizon, 7]
   - 含义: [x, y, z, qw, qx, qy, qz]
2. 船体速度时序
   - 形状: [batch, horizon, 6]
   - 含义: [vx, vy, vz, wx, wy, wz]
3. 船体加速度时序
   - 形状: [batch, horizon, 6]
   - 含义: [ax, ay, az, alphax, alphay, alphaz]
4. 可选: 直接外力时序
   - 形状: [batch, horizon, num_links, 6]

## 总体技术路线

1. 在状态转移层增加船体预测缓存与更新接口。
2. 在每次 compute_augmented_state 时，将当前预测转换成 f_ext。
3. 调用现有 compute_inverse_dynamics(state_seq, f_ext=...)。
4. 在 Rollout 与 Solver 层增加统一更新方法，支持每控制周期刷新。
5. 在任务配置中启用或提升现有 energy 权重。

## 分阶段实施计划

## 阶段 A：状态转移配置与运行时接口

### 1. 修改文件

- curobo/_src/transition/robot_state_transition_cfg.py
- curobo/_src/transition/robot_state_transition.py

### 2. 具体修改

#### curobo/_src/transition/robot_state_transition_cfg.py

新增配置字段（建议）：

1. enable_ship_motion_compensation: bool = False
2. ship_input_mode: str = "base_kinematics"
3. ship_apply_link: str = "base_link"
4. ship_force_gain: list[float]（长度 3）
5. ship_torque_gain: list[float]（长度 3）
6. ship_wrench_sign: float = 1.0

同时在 create 与 __post_init__ 中增加：

1. 默认值补齐。
2. 字段合法性校验。
3. 向后兼容旧配置。

#### curobo/_src/transition/robot_state_transition.py

新增成员与方法：

1. 船体预测缓存
   - 位姿缓存
   - 速度缓存
   - 加速度缓存
   - 直接外力缓存
2. 公开更新接口
   - set_ship_motion_prediction(...)
   - set_external_wrench_prediction(...)
   - clear_ship_motion_prediction()
3. 内部工具函数
   - 输入 shape 与 dtype 校验
   - 位姿和速度转换为基座扰动
   - 基座扰动映射到 f_ext

核心改动点：

1. compute_augmented_state 中调用逆动力学时传入 f_ext。
2. 无预测时保持当前行为（f_ext=None）。
3. 无 dynamics 时自动忽略补偿并保持原逻辑。

## 阶段 B：Rollout 与 Solver 统一分发

### 1. 修改文件

- curobo/_src/rollout/rollout_robot.py
- curobo/_src/solver/solver_core.py
- curobo/_src/solver/solver_mpc.py
- curobo/_src/solver/solver_trajopt.py
- curobo/_src/motion/motion_planner.py

### 2. 具体修改

#### curobo/_src/rollout/rollout_robot.py

新增方法 update_ship_motion_prediction(...):

1. 同步更新 transition_model。
2. 同步更新 metrics_transition_model。
3. 提供 clear 分支，便于循环中重置。

#### curobo/_src/solver/solver_core.py

新增方法 update_ship_motion_prediction(...):

1. 遍历 metrics_rollout、optimizer_rollouts、auxiliary_rollout、additional_metrics_rollouts。
2. 统一调用 rollout.update_ship_motion_prediction。

#### curobo/_src/solver/solver_mpc.py

新增对外方法 update_ship_motion_prediction(...)，直接透传到 core。

建议调用顺序：

1. update_goal_tool_poses 或 update_current_state。
2. update_ship_motion_prediction。
3. optimize_action_sequence。

#### curobo/_src/solver/solver_trajopt.py

新增对外方法 update_ship_motion_prediction(...)，透传到 core。

#### curobo/_src/motion/motion_planner.py

新增高层透传方法 update_ship_motion_prediction(...)，内部转发到 trajopt_solver。

## 阶段 C：配置层接入

### 1. 新增或修改文件

建议新增 ship 专用配置，避免污染现有默认配置：

- curobo/content/configs/robot/shipborne_ur10e.yml
- curobo/content/configs/scene/shipborne_calm.yml
- curobo/content/configs/scene/shipborne_rough.yml
- curobo/content/configs/task/mpc/shipborne_lbfgs_mpc.yml
- curobo/content/configs/task/mpc/shipborne_transition_bspline_mpc.yml
- curobo/content/configs/task/trajopt/shipborne_lbfgs_bspline_trajopt.yml
- curobo/content/configs/task/trajopt/shipborne_transition_bspline_trajopt.yml

### 2. 关键配置要求

1. 机器人配置必须启用动力学。
   - 明确设置 load_dynamics: true。
2. CSpace 必须使用 STATE 模式。
3. squared_l2_regularization_weight 第 5 项作为 energy 权重。
4. transition_model_cfg 中增加 ship 补偿开关与参数。

### 3. 第一版建议权重

1. calm 工况
   - energy 权重中等，优先可行性。
2. rough 工况
   - 适当提高 energy 与加速度正则项，抑制过激动作。

## 阶段 D：示例与演示

### 1. 新增示例文件（建议）

- curobo/examples/getting_started/shipborne_motion_planning.py
- curobo/examples/getting_started/shipborne_reactive_control.py

### 2. 示例内容

#### shipborne_motion_planning.py

1. 读取 ship 场景与 ship task 配置。
2. 注入一段离线船体预测。
3. 对比补偿开和关两次规划。
4. 输出能耗代价、最大扭矩、轨迹长度。

#### shipborne_reactive_control.py

1. 在控制循环每一周期更新预测。
2. 调用 update_ship_motion_prediction 后再 optimize_action_sequence。
3. 记录跟踪误差与控制平滑性。

## 阶段 E：测试与验收

### 1. 修改测试文件

- curobo/tests/_src/robot/dynamics/test_rnea_cuda.py
- curobo/tests/_src/transition/test_robot_state_transition.py
- curobo/tests/_src/solver/test_solver_mpc.py
- curobo/tests/_src/solver/test_solver_trajopt.py
- curobo/tests/_src/motion/test_motion_planner.py
- curobo/tests/_src/transition/test_robot_state_transition_cfg.py

### 2. 新增测试点

#### 动力学层

1. f_ext 为零与无 f_ext 行为一致。
2. f_ext 非零时扭矩变化符合预期。
3. f_ext 参与反向传播。

#### 状态转移层

1. 注入预测后 compute_augmented_state 输出扭矩改变。
2. batch 改变后缓存重建正确。
3. 清空预测后回归到原行为。

#### 求解器层

1. MPC 与 TrajOpt 的 update_ship_motion_prediction 接口可用。
2. 接口调用后优化流程无 shape 和 dtype 错误。

#### 高层接口

1. MotionPlanner 透传接口可用。
2. update_world 与 ship 更新可联合调用。

### 3. 验收指标

1. 可行率不低于基线。
2. 总能耗代价下降。
3. 峰值扭矩下降或不升高。
4. 轨迹平滑性不恶化。

## 逐文件修改清单（执行视图）

| 文件 | 修改类型 | 关键修改点 | 优先级 |
|---|---|---|---|
| curobo/_src/transition/robot_state_transition_cfg.py | Update | 新增 ship 补偿配置字段和校验 | P0 |
| curobo/_src/transition/robot_state_transition.py | Update | 缓存预测、构造 f_ext、接入 inverse dynamics | P0 |
| curobo/_src/rollout/rollout_robot.py | Update | 新增 rollout 级预测更新接口 | P0 |
| curobo/_src/solver/solver_core.py | Update | 新增统一分发接口 | P0 |
| curobo/_src/solver/solver_mpc.py | Update | 新增 MPC 对外更新接口 | P0 |
| curobo/_src/solver/solver_trajopt.py | Update | 新增 TrajOpt 对外更新接口 | P0 |
| curobo/_src/motion/motion_planner.py | Update | 新增高层透传接口 | P1 |
| curobo/content/configs/robot/shipborne_ur10e.yml | Add | ship 机器人配置和 load_dynamics | P0 |
| curobo/content/configs/scene/shipborne_calm.yml | Add | 平稳海况场景 | P1 |
| curobo/content/configs/scene/shipborne_rough.yml | Add | 扰动海况场景 | P1 |
| curobo/content/configs/task/mpc/shipborne_lbfgs_mpc.yml | Add | MPC ship 权重配置 | P0 |
| curobo/content/configs/task/mpc/shipborne_transition_bspline_mpc.yml | Add | MPC ship transition 配置 | P0 |
| curobo/content/configs/task/trajopt/shipborne_lbfgs_bspline_trajopt.yml | Add | TrajOpt ship 权重配置 | P0 |
| curobo/content/configs/task/trajopt/shipborne_transition_bspline_trajopt.yml | Add | TrajOpt ship transition 配置 | P0 |
| curobo/examples/getting_started/shipborne_motion_planning.py | Add | 离线对比示例 | P1 |
| curobo/examples/getting_started/shipborne_reactive_control.py | Add | 在线闭环示例 | P1 |
| curobo/tests/_src/robot/dynamics/test_rnea_cuda.py | Update | f_ext 前向和反向测试 | P0 |
| curobo/tests/_src/transition/test_robot_state_transition.py | Update | 扰动注入行为测试 | P0 |
| curobo/tests/_src/solver/test_solver_mpc.py | Update | MPC 接口联通测试 | P1 |
| curobo/tests/_src/solver/test_solver_trajopt.py | Update | TrajOpt 接口联通测试 | P1 |
| curobo/tests/_src/motion/test_motion_planner.py | Update | 高层透传与联合调用测试 | P1 |
| curobo/tests/_src/transition/test_robot_state_transition_cfg.py | Update | 新配置字段解析测试 | P1 |

## 风险与应对

1. 外力符号方向风险
   - 应对: 配置 ship_wrench_sign，并通过单测标定。
2. 输入 shape 风险
   - 应对: transition 层集中做校验与 reshape。
3. 回归风险
   - 应对: 默认关闭补偿，只有 ship 配置显式开启。
4. 性能风险
   - 应对: 第一版只做基座链路注入，不做全链高复杂映射。

## 里程碑与工期建议

1. M1（2-3 天）
   - 完成阶段 A 与阶段 B 核心接口。
   - 单元测试通过。
2. M2（2 天）
   - 完成配置与示例。
   - 完成离线与在线冒烟验证。
3. M3（1-2 天）
   - 完成参数整定与文档补充。
   - 形成可复现实验报告。

## 建议的实施顺序

1. 先改 transition cfg 和 transition 核心注入。
2. 再改 rollout 和 solver 分发接口。
3. 再补配置与示例。
4. 最后补测试并调参。

按此顺序执行，能最快得到一个可运行、可验证、可迭代的第一版船载机械臂扩展。