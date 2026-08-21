# 脚本08学习笔记：单张图片试管几何

对应文件：[`scripts_archive/08_infer_image_geometry.py`](../08_infer_image_geometry.py)

## 1. 这个脚本解决什么问题

对一张图片做分割，计算紫色/黄色试管的中心、B/C/G、抓取点、长度和角度。

## 2. 输入、输出和副作用

- 输入：单张彩色图、权重，以及可选旧仿射或外部 eye-to-hand 文件。
- 输出：终端/JSON 几何报告和可选可视化图片。
- 是否接触硬件：请结合下面的运行模式和注意事项判断；没有 `--execute` 不代表所有视觉脚本都不会打开相机。

## 3. 主流程

1. 读取并验证图像、配置和模型
2. 分颜色选择唯一 body/cap 实例
3. 从 mask 前景点做 PCA
4. 用 cap 消除主轴 180° 歧义
5. 计算抓取点和质量指标
6. 可选把像素射线与基座固定平面求交
7. 保存 JSON 与可视化

## 4. 建议阅读顺序

- `tube_pose_from_masks()`：二维 B/C/G 和抓取点
- `ExternalEyeToHandCalibration`：读取 `T_base_from_camera`
- `apply_eye_to_hand()`：把 B/C/G 像素投影到基座平面
- `main()`：串联单图推理和输出

初学者建议先读文件末尾的 `main()`，看清调用顺序后再回头阅读上面的辅助函数。

## 5. 本脚本涉及的 Python 语法

- dataclass 把多项几何结果组合为有名字的字段
- `np.asarray()` 统一输入为数组
- `@` 是矩阵乘法
- `None` 表示当前帧没有可声明的方向或坐标

## 6. 数学或领域知识

PCA 主轴来自协方差矩阵最大特征值。像素本身没有深度；eye-to-hand 模式用 `p_base=T_base_from_camera@p_camera` 和固定平面条件求射线交点。

## 7. 常用命令

\`\`\`bash
python scripts_archive/08_infer_image_geometry.py --image data/raw/你的会话/color/frame_000000.jpg --save-vis outputs/geometry.jpg
\`\`\`

\`\`\`bash
python scripts_archive/08_infer_image_geometry.py --image 图片路径 --eye-to-hand 标定文件 --output-json outputs/geometry.json
\`\`\`

脚本08固定使用 FP32 推理。

## 8. 容易混淆的地方

同色出现多个 body/cap 时脚本不会猜配对。`--allow-unvalidated-eye-to-hand` 只允许排查，输出不能直接用于运动。

## 9. 学习检查题

- 这个脚本的输入数据来自哪里？
- 哪一个函数产生最终结果？
- 坐标、长度和角度分别使用什么单位？
- 发生异常时，文件、相机或机械臂连接在哪里释放？
- 哪些检查只能降低风险，不能证明真实路径绝对安全？
