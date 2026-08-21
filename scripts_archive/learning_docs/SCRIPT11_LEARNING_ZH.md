# 脚本11学习笔记：图像方向转换为机械臂 yaw

对应文件：[`scripts_archive/11_preview_yellow_tube_robot_yaw.py`](../11_preview_yellow_tube_robot_yaw.py)

## 1. 这个脚本解决什么问题

把黄色试管从 B 到 C 的图像方向转换为机械臂基座平面 yaw 和两个等价夹爪四元数。

## 2. 输入、输出和副作用

- 输入：RealSense 实时检测，或 `--pixels center_u center_v B_u B_v C_u C_v`。
- 输出：中心基座坐标、B→C 基座 yaw、两个相差 180° 的候选四元数；绝不运动。
- 是否接触硬件：请结合下面的运行模式和注意事项判断；没有 `--execute` 不代表所有视觉脚本都不会打开相机。

## 3. 主流程

1. 检测唯一 y-body/y-cap 并计算 B/C
2. 检查多帧中心和方向稳定
3. 把 center/B/C 像素分别投影到基座平面
4. 用变换后的 B/C 重新计算基座 yaw
5. 叠加夹爪偏移并生成两个四元数
6. 打印预览报告

## 4. 建议阅读顺序

- `yellow_geometry()`：二维 center/B/C
- `pixel_to_arm()`：像素到基座平面
- `make_solution()`：基座 yaw 和两个候选
- `print_preview()`：明确标记 PREVIEW ONLY

初学者建议先读文件末尾的 `main()`，看清调用顺序后再回头阅读上面的辅助函数。

## 5. 本脚本涉及的 Python 语法

- `deque(maxlen=...)` 保存固定长度历史
- 圆周平均处理 ±180° 附近角度
- 元组保存不可变的两个候选
- 延迟导入 YOLO 和 RealSense

## 6. 数学或领域知识

图像角不能直接等于机械臂 yaw。必须把 B、C 两点变换到基座 XY 后计算 `atan2(C_y-B_y,C_x-B_x)`。对称夹爪沿轴旋转 180° 仍可夹取，因此有两个候选。

## 7. 常用命令

\`\`\`bash
python scripts_archive/11_preview_yellow_tube_robot_yaw.py --yaw-offset-deg 0
\`\`\`

\`\`\`bash
python scripts_archive/11_preview_yellow_tube_robot_yaw.py --pixels 720 119 674 50 767 188 --yaw-offset-deg 0
\`\`\`

脚本11固定使用 FP32 推理。

## 8. 容易混淆的地方

`yaw-offset-deg` 是夹爪安装参考方向偏移，不是试管随机摆放角；它应通过脚本12的高空实验确认。

## 9. 学习检查题

- 这个脚本的输入数据来自哪里？
- 哪一个函数产生最终结果？
- 坐标、长度和角度分别使用什么单位？
- 发生异常时，文件、相机或机械臂连接在哪里释放？
- 哪些检查只能降低风险，不能证明真实路径绝对安全？
