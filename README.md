# high-precision-multi-tube-manipulation

中文名：高精度多试管操作。

本项目使用 YOLO26 实例分割、Intel RealSense D435/D435i 和 CArm A3 机械臂，
完成试管数据采集、模型训练、实时分割预览，以及黄色/紫色试管连续抓取放置 demo。

当前工程采用“独立编号脚本工作流”：主要脚本都放在 `scripts/` 里，可以直接运行。
历史调试脚本已经先归档到`scripts_archive/`，没有直接删除，方便以后查参考。

公开仓库中可以保留代码、学习文档、数据集、原始采集样例、训练曲线和可视化结果；
但训练权重和模型导出文件不随仓库上传，例如 `*.pt`、`*.pth`、`*.onnx` 和
`runs/**/weights/`。需要权重时，请重新训练或从单独的模型发布位置下载。

> 重要：视觉预览、IK 检查和真机执行是不同阶段。IK/FK 通过不等于整条运动
> 路径无碰撞。任何 `--execute` 操作都需要清空扫掠区域、确认 tool=1、从规定
> 起始位开始，并准备好急停。

## 1. 当前主流程

```text
01 检查环境
  -> 03 RealSense 采集 RGB-D
  -> 04/05 转换为 YOLO Seg 数据集
  -> 06 训练 YOLO26 Seg
  -> 07 实时预览分割效果
  -> 15 单试管抓取放置 demo
  -> 17 黄色 + 紫色双试管连续 demo
```

四个模型类别及固定 ID：

| 类别 ID | 名称 | 含义 |
|---:|---|---|
| 0 | `p-body` | 紫色试管管身 |
| 1 | `p-cap` | 紫色试管管盖 |
| 2 | `y-body` | 黄色试管管身 |
| 3 | `y-cap` | 黄色试管管盖 |

类别顺序记录在 [classes.txt](configs/classes.txt)，脚本 07、15、17 内部也会
固定检查这四类顺序，不能随意重排。

## 2. 当前目录结构

```text
high-precision-multi-tube-manipulation/
├── configs/
│   └── classes.txt
├── data/raw/                       # RealSense 原始采集会话，可公开则上传
├── datasets/                       # YOLO Seg 数据集，可公开则上传
├── outputs/                        # 人工输出、预览图、JSON，可公开则上传
├── runs/                           # Ultralytics 训练和推理结果，排除 weights/
├── scripts/                        # 当前主流程脚本
├── scripts_archive/                # 历史调试脚本，暂时归档不删除
├── requirements.txt
├── requirements-yolo.txt
└── HPS_BEGINNER_SCRIPT_STYLE_SKILL_ZH.txt
```

当前没有 `src/`、`models/` 和 `tests/` 目录。工程先以独立脚本、脚本15/17
现场 demo 和 `py_compile` 静态检查为主；后续如果需要自动回归测试，再按当前
主流程重建 `tests/`。

## 3. Python 环境和包版本

以下是在本机 `hps` Conda 环境中实际读取到并通过检查的版本：

| 组件 | 当前版本 | 主要用途 |
|---|---:|---|
| Python | 3.10.20 | 运行全部编号脚本 |
| NumPy | 2.2.6 | 数组、矩阵、PCA、坐标变换 |
| SciPy | 1.15.2 | 数值计算依赖 |
| OpenCV | 4.13.0.92 | 图像、掩膜、相机窗口、绘图 |
| PyYAML | 6.0.3 | 读取训练、数据集和标定 YAML |
| tqdm | 4.67.3 | 进度显示依赖 |
| pyrealsense2 | 2.58.1.10581 | RealSense 彩色/RGB-D 数据流 |
| carm | 0.1.20260512 | CArm A3 Python SDK |
| PyTorch | 2.12.0+cu130 | CUDA 深度学习运行时 |
| torchvision | 0.27.0+cu130 | PyTorch 视觉组件 |
| Ultralytics | 8.4.56 | YOLO26 训练和实例分割推理 |

当前 PyTorch 实测状态：

```text
CUDA available: True
Torch CUDA: 13.0
GPU: NVIDIA GeForce RTX 3090
GPU count: 1
```

不要在 base Conda 环境中混装本项目 GPU 依赖，建议始终使用 `hps` 环境。

## 4. 环境安装与检查

```bash
conda activate hps
cd high-precision-multi-tube-manipulation
```

安装基础依赖：

```bash
python -m pip install -r requirements.txt
```

安装 GPU/YOLO 依赖：

```bash
python -m pip install -r requirements-yolo.txt
```

检查基础环境：

```bash
python scripts/01_check_environment.py
```

检查 YOLO、PyTorch 和 CUDA：

```bash
python scripts/01_check_environment.py --with-yolo
```

## 5. 当前主脚本

### 01：检查环境

```bash
python scripts/01_check_environment.py --with-yolo
```

只读检查，不安装软件、不打开相机、不连接机械臂。

### 03：RealSense RGB-D 数据采集

```bash
python scripts/03_collect_realsense_rgbd.py --out data/raw
```

运行后会新建时间戳会话目录，保存彩色图、深度图和相机内外参。

### 04：X-AnyLabeling 标注转换

```bash
python scripts/04_convert_xany_to_yolo_seg.py \
  --input data/raw/你的会话/color \
  --output datasets/tube_seg
```

输入目录中的图片必须有同名 JSON。确认路径后才使用 `--overwrite`。

### 05：Roboflow COCO 转换

```bash
python scripts/05_convert_roboflow_coco_to_yolo_seg.py \
  --input "/path/to/roboflow-coco-segmentation" \
  --output datasets/tube_seg_roboflow
```

如果不再使用 Roboflow 数据增强，这个脚本可以以后再归档。

### 06：训练 YOLO26 Seg

```bash
python scripts/06_train_yolo26_seg.py \
  --data datasets/tube_seg_roboflow/tube_seg.yaml \
  --model yolo26x-seg.pt \
  --epochs 100 \
  --imgsz 1024 \
  --batch 8 \
  --device 0 \
  --aug-profile roboflow-light
```

脚本06固定使用 FP32 训练；显存不足时先减小 `--batch`。

### 07：实时分割预览

```bash
python scripts/07_preview_realtime_seg.py \
  --source realsense
```

用于检查类别、置信度、掩膜边缘、实例数量和 FPS，不控制机械臂。

### 15：单试管抓取放置 demo

只读检查 IK：

```bash
python scripts/15_demo_single_tube.py \
  --serial YOUR_REALSENSE_SERIAL \
  --ip YOUR_CARM_IP \
  --check-ik
```

真实执行：

```bash
python scripts/15_demo_single_tube.py \
  --serial YOUR_REALSENSE_SERIAL \
  --ip YOUR_CARM_IP \
  --execute
```

### 17：黄色 + 紫色双试管连续 demo

只读检查 IK：

```bash
python scripts/17_demo_two_tube_branch_yaw.py \
  --serial YOUR_REALSENSE_SERIAL \
  --ip YOUR_CARM_IP \
  --check-ik
```

真实执行：

```bash
python scripts/17_demo_two_tube_branch_yaw.py \
  --serial YOUR_REALSENSE_SERIAL \
  --ip YOUR_CARM_IP \
  --execute
```

脚本17会一次锁定黄色和紫色试管，并按图像角度选择 POS/NEG 固定关节分支，然后：

```text
抓黄色 -> 放黄色凹槽 -> 回零
抓紫色 -> 放紫色凹槽 -> 回零
```

真实执行前只需要输入一次 `GRASP_TWO_TUBES ...` 确认文字。

## 6. 历史脚本

以下脚本已经先移动到 `scripts_archive/`，作为历史调试和学习参考：

```text
02_smoke_test_project.py
08_infer_image_geometry.py
09_preview_realsense_geometry.py
10_grasp_tube_demo.py
11_preview_yellow_tube_robot_yaw.py
12_test_gripper_yaw_alignment.py
13_grasp_yellow_tube_with_yaw.py
14_grasp_yellow_tube_cap_up.py
```

它们不是当前 demo 主线。当前主线优先使用脚本15和脚本17。

旧版脚本 08～14 的学习笔记已经移动到：

```text
scripts_archive/learning_docs/
```

当前主线学习建议优先看根目录的 `SCRIPT15_LEARNING_ZH.md`、
`SCRIPT17_LEARNING_ZH.md` 和 `SCRIPT17_MATH_PYTHON_ZH.md`。

## 7. 配置文件说明

### `configs/classes.txt`

脚本04和05读取它决定 YOLO 类别 ID。四行顺序必须固定。

当前 `configs/` 中只保留 `classes.txt` 这一个确实有消费者的文件。

## 8. 测试说明

当前没有 `tests/` 目录。现在最实用的轻量检查是语法编译：

```bash
python -m py_compile scripts/*.py
python -m py_compile scripts_archive/*.py
```

这些检查不会打开 RealSense，也不会连接或驱动机械臂。真实相机、模型推理和
CArm 运动仍需要用对应脚本单独验证。

## 9. 常见提醒

- 图像角不能直接当机械臂 yaw，需要通过手眼标定转换到基座坐标系。
- `YOUR_REALSENSE_SERIAL` 和 `YOUR_CARM_IP` 需要替换成自己的相机序列号和机械臂 IP。
- `--check-ik` 只读检查，不使能、不运动。
- `--execute` 才是真机执行，必须确认扫掠区域、夹爪、试管和急停。
- OpenCV 的 `QFontDatabase` 字体警告通常不影响分割推理和窗口显示。
