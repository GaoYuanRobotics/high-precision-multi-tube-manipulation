# 脚本01学习笔记：检查 Python 环境

对应文件：[`scripts/01_check_environment.py`](scripts/01_check_environment.py)

## 1. 这个脚本解决什么问题

确认当前解释器能导入项目基础包；使用 `--with-yolo` 时额外核对 PyTorch、Ultralytics 和 CUDA。

## 2. 输入、输出和副作用

- 输入：当前激活的 Python/Conda 环境，以及可选参数 `--with-yolo`。
- 输出：终端中的 `[ok]`、`[missing]`、版本不匹配和 CUDA 状态；不会修改环境。
- 是否接触硬件：请结合下面的运行模式和注意事项判断；没有 `--execute` 不代表所有视觉脚本都不会打开相机。

## 3. 主流程

1. 读取 `CORE_MODULES` 和 `YOLO_MODULES` 依赖契约
2. 用 `importlib.import_module()` 真实导入每个模块
3. 用 `importlib.metadata.version()` 读取 pip 发行包版本
4. 汇总缺失项；有问题返回退出码 1，否则返回 0

## 4. 建议阅读顺序

- `check_packages()`：逐个导入并核对固定版本
- `installed_version()`：读取安装版本
- `print_cuda_status()`：只在要求 YOLO 检查时延迟导入 torch
- `main()`：决定检查基础依赖还是完整 GPU 依赖

初学者建议先读文件末尾的 `main()`，看清调用顺序后再回头阅读上面的辅助函数。

## 5. 本脚本涉及的 Python 语法

- 列表中的三元组解包：`module_name, package_name, required_version`
- `None` 表示不固定精确版本
- 异常捕获用于把二进制库导入失败变成可读报告
- 退出码 0/1 可供终端和自动测试判断成功失败

## 6. 数学或领域知识

本脚本没有几何公式。重点是区分“pip 包名”和“Python 导入名”，例如 `opencv-python` 的导入名是 `cv2`。

## 7. 常用命令

\`\`\`bash
python scripts/01_check_environment.py
\`\`\`

\`\`\`bash
python scripts/01_check_environment.py --with-yolo
\`\`\`

## 8. 容易混淆的地方

`torch.cuda.is_available()` 为 True 才表示当前 PyTorch 能使用 CUDA；仅仅安装了驱动或看到 GPU 名称还不够。

## 9. 学习检查题

- 这个脚本的输入数据来自哪里？
- 哪一个函数产生最终结果？
- 坐标、长度和角度分别使用什么单位？
- 发生异常时，文件、相机或机械臂连接在哪里释放？
- 哪些检查只能降低风险，不能证明真实路径绝对安全？
