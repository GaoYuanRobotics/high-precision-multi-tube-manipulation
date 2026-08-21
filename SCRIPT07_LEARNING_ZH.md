# 脚本07学习笔记：实时分割预览

对应文件：[`scripts/07_preview_realtime_seg.py`](scripts/07_preview_realtime_seg.py)

## 1. 这个脚本解决什么问题

从 RealSense、USB 相机、视频或图片读取画面，实时显示四类分割掩膜和 FPS。

## 2. 输入、输出和副作用

- 输入：YOLO Seg 权重、视频源和推理参数。
- 输出：OpenCV 预览窗口，以及可选截图、最后一帧图片或视频。
- 是否接触硬件：请结合下面的运行模式和注意事项判断；没有 `--execute` 不代表所有视觉脚本都不会打开相机。

## 3. 主流程

1. 严格读取视觉配置和四类顺序
2. 选择模型权重并核对 segment 任务
3. 根据 source 创建对应帧源
4. 逐帧执行 `model.predict()`
5. 叠加掩膜、类别、置信度、计数和 FPS
6. 按键退出并在 finally 中释放资源

## 4. 建议阅读顺序

- `validate_tube_model_contract()`：检查模型必须是四类实例分割模型
- `validate_tube_model_contract()`：拒绝类别顺序错误的权重
- `RealSenseColorSource`：管理相机生命周期
- `main()`：统一图片、视频和相机预览循环

初学者建议先读文件末尾的 `main()`，看清调用顺序后再回头阅读上面的辅助函数。

## 5. 本脚本涉及的 Python 语法

- 类适合保存 pipeline 这类有生命周期的状态
- `try/finally` 保证相机和窗口释放
- 指数移动平均平滑 FPS
- `argparse.BooleanOptionalAction` 同时生成开启/关闭选项

## 6. 数学或领域知识

FPS 使用相邻帧的单调时钟间隔计算；置信度和 IoU 是不同阈值，前者筛实例，后者影响 NMS 重叠抑制。

## 7. 常用命令

\`\`\`bash
python scripts/07_preview_realtime_seg.py --source realsense
\`\`\`

\`\`\`bash
python scripts/07_preview_realtime_seg.py --source 0
\`\`\`

\`\`\`bash
python scripts/07_preview_realtime_seg.py --source 图片路径 --no-display --max-frames 1 --save-vis outputs/preview.jpg
\`\`\`

脚本07固定使用 FP32 推理。

## 8. 容易混淆的地方

脚本07只检查分割效果，不计算机械臂坐标，也不会连接机械臂。多台 RealSense 时必须明确 `--serial`。

## 9. 学习检查题

- 这个脚本的输入数据来自哪里？
- 哪一个函数产生最终结果？
- 坐标、长度和角度分别使用什么单位？
- 发生异常时，文件、相机或机械臂连接在哪里释放？
- 哪些检查只能降低风险，不能证明真实路径绝对安全？
