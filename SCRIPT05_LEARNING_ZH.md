# 脚本05学习笔记：Roboflow COCO 转 YOLO Seg

对应文件：[`scripts/05_convert_roboflow_coco_to_yolo_seg.py`](scripts/05_convert_roboflow_coco_to_yolo_seg.py)

## 1. 这个脚本解决什么问题

把 Roboflow 导出的 COCO Segmentation 数据转换为 YOLO Seg，并按原图来源分组划分。

## 2. 输入、输出和副作用

- 输入：Roboflow 根目录、train 目录或 `_annotations.coco.json`。
- 输出：`datasets/tube_seg_roboflow`、数据集 YAML 和 `conversion_report.json`。
- 是否接触硬件：请结合下面的运行模式和注意事项判断；没有 `--execute` 不代表所有视觉脚本都不会打开相机。

## 3. 主流程

1. 定位 COCO JSON 和图片目录
2. 严格读取 categories/images/annotations
3. 把 COCO 类别 ID 映射到四类 YOLO ID
4. 用 `images[].extra.name` 找到增强图片的原图组
5. 按原图组划分 train/val，防止同源泄漏
6. 转换 polygon、写报告并验证输出

## 4. 建议阅读顺序

- `find_coco_annotation()`：定位 COCO 文件
- `_prepare_images()`：校验并整理每张图片
- `_split_source_groups()`：按原图组划分
- `convert_roboflow_coco_dataset()`：完整转换流程

初学者建议先读文件末尾的 `main()`，看清调用顺序后再回头阅读上面的辅助函数。

## 5. 本脚本涉及的 Python 语法

- `defaultdict(list)` 自动为新键创建列表
- `Counter` 统计各类别实例数
- 普通 `for` 循环逐项生成报告字段
- `asdict()` 把 dataclass 转为可写入 JSON 的字典

## 6. 数学或领域知识

关键不是随机分图片，而是随机分“原图组”。同一原图的旋转、亮度等增强版本必须全部在 train 或全部在 val。

## 7. 常用命令

\`\`\`bash
python scripts/05_convert_roboflow_coco_to_yolo_seg.py --input '/home/gaoyuan/下载/New HPS.v2-new-version.coco-segmentation'
\`\`\`

\`\`\`bash
python scripts/05_convert_roboflow_coco_to_yolo_seg.py --input 你的Roboflow目录 --overwrite
\`\`\`

## 8. 容易混淆的地方

转换报告应重点检查 train/val 原图组数量、四类标签计数和 ignored categories。

## 9. 学习检查题

- 这个脚本的输入数据来自哪里？
- 哪一个函数产生最终结果？
- 坐标、长度和角度分别使用什么单位？
- 发生异常时，文件、相机或机械臂连接在哪里释放？
- 哪些检查只能降低风险，不能证明真实路径绝对安全？
