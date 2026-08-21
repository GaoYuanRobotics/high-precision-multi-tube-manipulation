# 脚本13学习笔记：动态 yaw 黄色试管抓取

对应文件：[`scripts_archive/13_grasp_yellow_tube_with_yaw.py`](../13_grasp_yellow_tube_with_yaw.py)

## 1. 这个脚本解决什么问题

把脚本11/12验证的动态 yaw 合并进完整流程：RealSense 检测、基座定位、IK 选择、下降闭爪和抬高。

## 2. 输入、输出和副作用

- 输入：RealSense、四类模型、偏移角、CArm IP 和可选 `--execute`。
- 输出：默认只预览目标和计划；execute 会真实抓取并在高位保持闭爪。
- 是否接触硬件：请结合下面的运行模式和注意事项判断；没有 `--execute` 不代表所有视觉脚本都不会打开相机。

## 3. 主流程

1. 检测稳定的 center/B/C
2. 把三点转换到基座平面
3. 生成两个等价夹爪 yaw
4. 连接 CArm 后对两个候选做 IK/FK
5. 选择准备位关节变化较小的候选
6. 准备位→高位 yaw→下降→等待→闭爪→抬高

## 4. 建议阅读顺序

- `detect_and_lock()`：实时检测和锁定
- `make_solution()`：基座 XY/yaw
- `choose_yaw_by_ik()`：比较两个候选
- `execute_grasp()`：真机动作序列
- `CarmClient`：脚本内置的 SDK 包装

初学者建议先读文件末尾的 `main()`，看清调用顺序后再回头阅读上面的辅助函数。

## 5. 本脚本涉及的 Python 语法

- dataclass 管理 Pose 和 CArm 配置
- 属性 `@property` 隐藏未连接状态检查
- 上下文管理器自动 disconnect
- 函数注解明确米、弧度和可选值

## 6. 数学或领域知识

动态 yaw 通过四元数表达，候选方向相差 π。姿态误差使用四元数点积得到最小旋转角，且正确处理 q 与 -q 表示同一旋转。

## 7. 常用命令

\`\`\`bash
python scripts_archive/13_grasp_yellow_tube_with_yaw.py --yaw-offset-deg 0
\`\`\`

\`\`\`bash
python scripts_archive/13_grasp_yellow_tube_with_yaw.py --yaw-offset-deg 0 --ip 10.42.0.101 --execute
\`\`\`

脚本13固定使用 FP32 推理。

## 8. 容易混淆的地方

脚本13抓取后只抬高悬停，不负责盖朝上、插槽、松爪或回零。真机前先用11确认坐标、用12确认高空 yaw。

## 9. 学习检查题

- 这个脚本的输入数据来自哪里？
- 哪一个函数产生最终结果？
- 坐标、长度和角度分别使用什么单位？
- 发生异常时，文件、相机或机械臂连接在哪里释放？
- 哪些检查只能降低风险，不能证明真实路径绝对安全？
