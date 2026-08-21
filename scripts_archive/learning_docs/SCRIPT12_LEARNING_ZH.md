# 脚本12学习笔记：高空夹爪 yaw 对齐测试

对应文件：[`scripts_archive/12_test_gripper_yaw_alignment.py`](../12_test_gripper_yaw_alignment.py)

## 1. 这个脚本解决什么问题

把脚本11输出的 XY/yaw 放到安全高处做 IK/FK 和可选真机对齐，不下降、不闭爪。

## 2. 输入、输出和副作用

- 输入：`--x`、`--y`、`--tube-yaw-deg`、偏移角、高度和 CArm IP。
- 输出：默认只打印计划；execute 模式执行“零位→准备位→高空对齐→观察→准备位→零位”。
- 是否接触硬件：请结合下面的运行模式和注意事项判断；没有 `--execute` 不代表所有视觉脚本都不会打开相机。

## 3. 主流程

1. 根据试管 yaw 生成两个相差 180° 的 Pose
2. 打印 dry-run 计划
3. execute 时只读连接并检查两个候选 IK/FK
4. 选择关节变化较小且通过限制的候选
5. 再次确认后使能并执行高空动作
6. 观察后自动回零

## 4. 建议阅读顺序

- `make_candidate_poses()`：构造两个四元数候选
- `evaluate_candidate()`：IK/FK 与误差检查
- `execute_alignment()`：高空真机流程
- `validate_args()`：限制高度和数值范围

初学者建议先读文件末尾的 `main()`，看清调用顺序后再回头阅读上面的辅助函数。

## 5. 本脚本涉及的 Python 语法

- 命令行必填参数使用 `required=True`
- 枚举候选并记录失败原因
- 回调线程错误通过共享状态传回主线程
- 角度显示用 degrees，内部计算仍用 radians

## 6. 数学或领域知识

IK 从 Pose 求关节角，FK 从关节角反算 Pose。检查 FK 误差可以发现控制器返回解与目标不一致；但 IK/FK 通过仍不等于整条路径已做碰撞规划。

## 7. 常用命令

\`\`\`bash
python scripts_archive/12_test_gripper_yaw_alignment.py --x 0.253031 --y -0.004325 --tube-yaw-deg -1.532 --yaw-offset-deg 0 --align-z 0.300
\`\`\`

\`\`\`bash
python scripts_archive/12_test_gripper_yaw_alignment.py --x 0.253031 --y -0.004325 --tube-yaw-deg -1.532 --yaw-offset-deg 0 --align-z 0.300 --ip 10.42.0.101 --execute
\`\`\`

## 8. 容易混淆的地方

脚本默认偏移值以 `--help` 为准；现场确认是 0° 时请明确传 `--yaw-offset-deg 0`，不要依赖默认值。

## 9. 学习检查题

- 这个脚本的输入数据来自哪里？
- 哪一个函数产生最终结果？
- 坐标、长度和角度分别使用什么单位？
- 发生异常时，文件、相机或机械臂连接在哪里释放？
- 哪些检查只能降低风险，不能证明真实路径绝对安全？
