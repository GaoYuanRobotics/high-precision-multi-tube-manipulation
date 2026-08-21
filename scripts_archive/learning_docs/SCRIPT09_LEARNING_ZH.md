# 脚本09学习笔记：RealSense 实时几何预览

对应文件：[`scripts_archive/09_preview_realsense_geometry.py`](../09_preview_realsense_geometry.py)

## 1. 这个脚本解决什么问题

把脚本08的单图几何扩展到 RealSense 连续彩色流，实时显示 B/C/G 和可选基座坐标。

## 2. 输入、输出和副作用

- 输入：RealSense、YOLO 权重、视觉配置和可选 eye-to-hand 标定。
- 输出：实时窗口、状态面板、截图和可选最后一帧图片。
- 是否接触硬件：请结合下面的运行模式和注意事项判断；没有 `--execute` 不代表所有视觉脚本都不会打开相机。

## 3. 主流程

1. 严格选择 RealSense 序列号
2. 启动并核对实际彩色流
3. 逐帧分割四类实例
4. 计算每种颜色的二维姿态和质量状态
5. 可选转换到机械臂基座固定平面
6. 绘制结果并处理按键
7. 退出时停止 pipeline

## 4. 建议阅读顺序

- `RealSenseColorSource`：相机连接、内参和帧读取
- `calculate_frame_geometries()`：一帧的两种颜色几何
- `apply_eye_to_hand()`：实时基座坐标显示
- `draw_status_panel()`：把失败原因和质量指标画在屏幕上

初学者建议先读文件末尾的 `main()`，看清调用顺序后再回头阅读上面的辅助函数。

## 5. 本脚本涉及的 Python 语法

- `@dataclass(frozen=True)` 防止帧结果被意外修改
- 字典按 `purple/yellow` 保存两个独立结果
- 上下文中的 `Any` 用于兼容 Ultralytics 动态对象
- 资源清理必须放在 `finally`

## 6. 数学或领域知识

图像角是相对图像 x 轴的 `atan2(dy,dx)`；基座 yaw 必须先把 B/C 分别变换到同一基座平面后重新计算，不能直接把图像角交给机械臂。

## 7. 常用命令

\`\`\`bash
python scripts_archive/09_preview_realsense_geometry.py
\`\`\`

\`\`\`bash
python scripts_archive/09_preview_realsense_geometry.py --serial 你的序列号
\`\`\`

\`\`\`bash
python scripts_archive/09_preview_realsense_geometry.py --no-display --max-frames 30 --save-vis outputs/realtime_geometry.jpg
\`\`\`

脚本09固定使用 FP32 推理。

## 8. 容易混淆的地方

脚本09只显示坐标，不控制机械臂；实时标定身份、分辨率和相机光学流必须与外参记录一致。

## 9. 学习检查题

- 这个脚本的输入数据来自哪里？
- 哪一个函数产生最终结果？
- 坐标、长度和角度分别使用什么单位？
- 发生异常时，文件、相机或机械臂连接在哪里释放？
- 哪些检查只能降低风险，不能证明真实路径绝对安全？
