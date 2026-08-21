# 脚本04学习笔记：X-AnyLabeling 转 YOLO Seg

对应文件：[`scripts/04_convert_xany_to_yolo_seg.py`](scripts/04_convert_xany_to_yolo_seg.py)

## 1. 这个脚本解决什么问题

把图片旁边的 X-AnyLabeling/LabelMe polygon JSON 转成 Ultralytics YOLO 分割目录。

## 2. 输入、输出和副作用

- 输入：包含图片和同名 `.json` 的目录、`configs/classes.txt`、验证集比例和随机种子。
- 输出：`images/train|val`、`labels/train|val` 和 `tube_seg.yaml`。
- 是否接触硬件：请结合下面的运行模式和注意事项判断；没有 `--execute` 不代表所有视觉脚本都不会打开相机。

## 3. 主流程

1. 读取类别顺序并建立类别 ID
2. 递归配对图片和同名 JSON
3. 校验尺寸、类别、polygon 点和边界
4. 把像素坐标除以宽高归一化到 0..1
5. 随机划分 train/val 并在临时目录生成
6. 完整复查后发布输出目录

## 4. 建议阅读顺序

- `convert_shape_to_yolo_line()`：一条 polygon 转一行 YOLO 标签
- `convert_one_json()`：转换一张图片的全部实例
- `convert_dataset()`：组织整个数据集
- `_validate_generated_dataset()`：发布前复核图片/标签和 polygon

初学者建议先读文件末尾的 `main()`，看清调用顺序后再回头阅读上面的辅助函数。

## 5. 本脚本涉及的 Python 语法

- `@dataclass(frozen=True)` 保存不可变统计结果
- `Path.rglob()` 递归寻找文件
- 集合用于比较图片 stem 和标签 stem
- `try/except/finally` 确保失败时删除临时目录

## 6. 数学或领域知识

YOLO 坐标为 `x_norm=x_px/width`、`y_norm=y_px/height`。鞋带公式检查 polygon 面积，面积为 0 的退化多边形必须拒绝。

## 7. 常用命令

\`\`\`bash
python scripts/04_convert_xany_to_yolo_seg.py --input data/raw/你的会话/color
\`\`\`

\`\`\`bash
python scripts/04_convert_xany_to_yolo_seg.py --input data/raw/你的会话/color --output datasets/tube_seg --overwrite
\`\`\`

## 8. 容易混淆的地方

`classes.txt` 行顺序就是类别 ID，不能随意重排。`--overwrite` 会替换已有输出，使用前先核对路径。

## 9. 学习检查题

- 这个脚本的输入数据来自哪里？
- 哪一个函数产生最终结果？
- 坐标、长度和角度分别使用什么单位？
- 发生异常时，文件、相机或机械臂连接在哪里释放？
- 哪些检查只能降低风险，不能证明真实路径绝对安全？
