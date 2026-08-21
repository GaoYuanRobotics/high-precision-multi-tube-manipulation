# 脚本 03 学习笔记：RealSense RGB-D 数据采集

对应脚本：`scripts/03_collect_realsense_rgbd.py`

脚本 03 的任务很明确：从一台身份确定的 RealSense 相机读取彩色帧和深度帧，
在用户按 `S` 时把同一编号的数据完整保存到一个新会话中。

## 一、推荐阅读顺序

1. `main()`：先看完整流程。
2. `validate_args()`：看会话目录如何避免覆盖旧数据。
3. `RealSenseCapture`：看相机如何选择、启动和关闭。
4. `capture_loop()`：看预览、按键和保存时机。
5. `save_frame()`：最后学习多文件安全保存。

总体数据流：

```text
命令行参数
  -> 验证会话目录
  -> 枚举并选择唯一相机
  -> 启动彩色/深度流
  -> 深度对齐到彩色图
  -> 实时预览
  -> 按 S 或定时触发
  -> 临时写入 JPEG/PNG/NPY/JSON
  -> 全部成功后发布最终文件
```

## 二、运行方式

默认采集 RGB-D：

```bash
python scripts/03_collect_realsense_rgbd.py --out data/raw
```

只采集彩色图：

```bash
python scripts/03_collect_realsense_rgbd.py \
  --out data/raw \
  --no-depth
```

指定相机和自动保存间隔：

```bash
python scripts/03_collect_realsense_rgbd.py \
  --out data/raw \
  --serial 135122076361 \
  --auto-interval 2.0
```

- 按 `S`：保存当前完整帧。
- 按 `Q`：退出采集。
- 多台 RealSense 同时连接时必须提供 `--serial`。

## 三、输出文件

一次运行只创建一个全新会话：

```text
data/raw/20260804_120000/
├── color/frame_000000.jpg
├── depth/frame_000000.png
├── depth_npy/frame_000000.npy
├── meta/frame_000000.json
├── intrinsics.json
└── session.json
```

各文件含义：

| 文件 | 内容 |
|---|---|
| `color/*.jpg` | 用于查看和标注的 BGR 彩色图 |
| `depth/*.png` | 对齐到彩色图的原始深度整数，无损 16 位 PNG |
| `depth_npy/*.npy` | 同一深度数组，保留 NumPy 形状和数据类型 |
| `meta/*.json` | 保存编号、系统时间、彩色/深度帧号 |
| `intrinsics.json` | 彩色/深度内参、外参、深度单位 |
| `session.json` | 相机序列号、固件和实际流配置 |

真实米制深度为：

```text
depth_m = depth_integer * depth_scale_m
```

## 四、Python 语法

### 1. `Path`

```python
session_dir / "color" / "frame_000000.jpg"
```

斜杠在 `Path` 对象之间表示拼接路径，比手工拼接字符串更清楚。

### 2. `str | None`

```python
requested_serial: str | None
```

表示序列号可以是字符串，也可以是没有提供时的 `None`。

### 3. 延迟导入

```python
class RealSenseCapture:
    def __init__(self, args, session_dir):
        import pyrealsense2 as rs
```

只有真正创建相机对象时才加载 SDK。因此：

- `--help` 不会打开相机；
- 导入脚本做文件保存测试时不会枚举设备；
- 相机相关操作集中在一个清楚位置。

### 4. 类和状态

`RealSenseCapture` 保存同一次会话共享的状态：

- `pipeline`
- `selected_serial`
- `device_serial`
- `align`
- 是否已经启动

这是适合使用类的场景，因为这些数据从 `start()` 一直使用到 `close()`。

### 5. `try/finally`

```python
camera = RealSenseCapture(args, session_dir)
try:
    camera.start()
    capture_loop(camera, args)
finally:
    camera.close()
```

无论用户正常退出还是中途发生异常，`finally` 都会关闭相机和 OpenCV 窗口。

### 6. 字典

JSON 数据先写成 Python 字典：

```python
metadata = {
    "index": saved_count,
    "timestamp_unix_s": time.time(),
}
```

字典的键会成为 JSON 字段名。

### 7. 普通 if/else

```python
depth = None
if depth_frame:
    depth = np.asanyarray(depth_frame.get_data())
```

有深度帧就转换成数组，否则返回 `None`。

### 8. `continue` 和 `break`

- `continue`：当前帧不完整，跳到下一次循环。
- `break`：用户按 `Q`，结束整个采集循环。

## 五、RealSense 基础知识

### 1. 彩色流和深度流

彩色流使用 `bgr8`，通道顺序与 OpenCV 一致。

深度流使用 `z16`，每个像素是 16 位无符号整数。这个整数不是直接的米，必须
乘以 `depth_scale_m`。

### 2. 为什么要深度对齐

RealSense 的彩色相机和深度相机位于不同物理位置，原始图像网格并不完全一致。

```python
align = rs.align(rs.stream.color)
frames = align.process(frames)
```

这一步把深度重新投影到彩色光学坐标系。保存后，相同 `(u,v)` 可以对应彩色
像素和对齐深度像素。

### 3. 相机内参

```text
fx, fy   焦距，单位为像素
ppx, ppy 主点像素坐标
coeffs   镜头畸变系数
```

内参必须和分辨率一致。改变 1280×720 流配置后，不能继续假设旧内参仍适用。

### 4. 深度到彩色外参

外参包含：

- 3×3 旋转矩阵；
- 3×1 平移向量，单位为米。

librealsense 返回的九元素旋转数组采用列主序。脚本使用：

```python
rotation_matrix = rotation_flat.reshape((3, 3), order="F")
```

把它还原成通常按行阅读的数学矩阵，并同时保留 SDK 原始排列便于检查。

## 六、为什么严格选择相机

如果电脑只连接一台相机，脚本可以使用唯一设备。

如果连接多台相机而没有传 `--serial`，脚本会停止。原因是：

- 手眼标定只对应某一台具体相机；
- 不同相机内参不同；
- 自动使用“第一台设备”可能导致采集身份与标定不匹配。

相机启动后还会再次读取实际序列号，确认 SDK 没有启动另一台设备。

## 七、为什么不复用旧会话目录

```python
session_dir.mkdir(exist_ok=False)
```

目标目录已经存在就失败，即使目录为空也不接管。这样可以避免：

- 新旧采集混在一起；
- `frame_000000` 覆盖旧文件；
- 两个采集进程同时写同一会话；
- 接管上次崩溃留下的不完整目录。

## 八、单帧安全保存

`save_frame()` 不直接把四个最终文件逐个写出去，而是：

1. 检查所有最终文件都不存在。
2. 为本帧创建随机临时目录。
3. 把 JPEG、PNG、NPY、JSON 全部写入临时目录。
4. 所有编码成功后，才发布最终文件名。
5. 发布中途失败时，回滚本次已经发布的文件。
6. 删除临时目录。

### 单文件原子发布

```python
os.link(staged_path, final_path)
```

硬链接创建最终文件名时不会覆盖已有文件；同一文件系统中的单个名字发布是原子
操作。

### 不是多文件事务

JPEG、PNG、NPY 和 JSON 仍然需要逐个发布。因此必须准确理解：

- 普通 Python 异常会触发回滚；
- 进程被强制杀死或机器断电后，仍可能留下不完整的一组文件；
- 不能把这段代码描述为“整个 RGB-D 帧一次性原子提交”。

这也是测试中同时检查成功保存和失败回滚的原因。

## 九、自动保存时间

```python
last_auto_save = time.monotonic()
```

自动间隔使用单调时钟，而不是系统日期时间。即使系统时钟被同步或人工修改，
单调时钟仍然只向前走，适合计算时间差。

逐帧元数据另外使用：

```python
time.time()
```

它生成 Unix 时间戳，适合与其他设备日志对时。

## 十、当前边界

- 自动化测试不打开真实 RealSense。
- RGB-D 对齐质量仍需要用真实相机画面检查。
- JPEG 是有损格式，适合标注；深度 PNG 和 NPY 是无损数据。
- `--no-depth` 模式仍创建空的 `depth`、`depth_npy` 目录，保持会话结构统一。
- 相机位置、设备、流配置或外部标定改变后，必须重新核对匹配关系。
- 如果操作系统或存储设备突然断电，应检查会话中每个编号的文件是否成套。

先掌握 `main()`、`RealSenseCapture` 和 `capture_loop()`，再研究外参排列与安全
写入，会比从第一行顺序阅读容易得多。
