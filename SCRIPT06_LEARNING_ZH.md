# 脚本06学习笔记：训练 YOLO26 分割模型

对应文件：[`scripts/06_train_yolo26_seg.py`](scripts/06_train_yolo26_seg.py)

## 1. 这个脚本解决什么问题

验证数据集契约后，用 Ultralytics 训练四类 YOLO26 实例分割模型。

## 2. 输入、输出和副作用

- 输入：数据集 YAML、初始 `.pt` 权重、轮数、图像尺寸、batch、GPU 和增强档位。
- 输出：`runs/segment/...` 下的权重、指标曲线、参数和日志。
- 是否接触硬件：请结合下面的运行模式和注意事项判断；没有 `--execute` 不代表所有视觉脚本都不会打开相机。

## 3. 主流程

1. 解析训练参数
2. 严格读取数据集 YAML
3. 核对四类名称、train/val 目录和标签
4. 统计每个 split 的类别实例
5. 根据增强档位准备训练参数
6. 延迟导入 YOLO 并调用 `model.train()`

## 4. 建议阅读顺序

- `validate_training_inputs()`：训练前完整数据检查
- `parse_args()`：定义训练选项
- `train_model()`：构造并提交 Ultralytics 参数
- `main()`：预检通过后才开始训练

初学者建议先读文件末尾的 `main()`，看清调用顺序后再回头阅读上面的辅助函数。

## 5. 本脚本涉及的 Python 语法

- `choices=(...)` 限制命令行只能选择三档增强
- 从增强字典逐项读取 `degrees`、`translate` 等参数，再明确传给 `model.train()`
- `yaml.load()` 使用拒绝重复键的 Loader
- `Path.resolve()` 统一数据路径

## 6. 数学或领域知识

训练指标可关注 mask mAP50、mAP50-95、precision 和 recall。在线增强不是越强越好：Roboflow 已增强数据默认使用 `roboflow-light`。

## 7. 常用命令

\`\`\`bash
python scripts/06_train_yolo26_seg.py
\`\`\`

\`\`\`bash
python scripts/06_train_yolo26_seg.py --data datasets/tube_seg_roboflow/tube_seg.yaml --aug-profile roboflow-light
\`\`\`

\`\`\`bash
python scripts/06_train_yolo26_seg.py --epochs 100 --imgsz 1024 --batch 8 --device 0
\`\`\`

## 8. 容易混淆的地方

本脚本通过 `amp=False` 固定使用 FP32 训练，不启用 AMP 混合精度，因此显存占用会
高于混合精度训练。本脚本会真实使用 GPU 并长时间写入 runs；先运行脚本01的
`--with-yolo`，显存不足时先减小 batch。

## 9. 学习检查题

- 这个脚本的输入数据来自哪里？
- 哪一个函数产生最终结果？
- 坐标、长度和角度分别使用什么单位？
- 发生异常时，文件、相机或机械臂连接在哪里释放？
- 哪些检查只能降低风险，不能证明真实路径绝对安全？
