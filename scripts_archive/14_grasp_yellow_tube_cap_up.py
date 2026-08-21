#!/usr/bin/env python3
"""最简单的黄色试管抓取并转成盖子朝上 Demo。

流程只有七件事：

1. RealSense + YOLO 找到 y-body 和 y-cap；
2. 手眼矩阵把试管中心和方向变换到机械臂基座；
3. CArm 只读计算一条可达的 IK 路径；
4. 对齐试管 yaw，下降并闭爪；
5. 抬高后分 45°、90° 两段把盖子转到 +Z；
6. 保持盖朝上，依次移动到固定凹槽高空位和释放位；
7. 到达新的释放位后暂停，等待后续现场调整。

默认只做视觉预览。``--check-ik`` 只读检查路径，``--execute`` 才真实运动。
本文件包含完整流程，适合从 ``main()`` 开始阅读。
配套学习笔记见工程根目录的 ``SCRIPT14_LEARNING_ZH.md``。
"""

from __future__ import annotations

# argparse：读取 --check-ik、--execute 等命令行参数。
# math：三角函数、角度/弧度转换；time：等待和超时计时。
import argparse
import math
import time

# deque 是固定长度的双端队列，本脚本用它保存最近 10 帧检测结果。
from collections import deque

# Path 用面向对象的方式拼接文件路径；Any、Sequence 只用于类型提示。
from pathlib import Path
from typing import Any, Sequence

# cv2 负责图像、掩膜和窗口；NumPy 负责向量/矩阵；yaml 读取相机配置。
import cv2
import numpy as np
import yaml


# =============================================================================
# 1. 现场已经确认过的固定参数
# =============================================================================

# __file__ 是当前脚本路径；parents[1] 表示向上两级得到工程根目录。
ROOT = Path(__file__).resolve().parents[1]
# 手眼标定由另一个工程完成，这里只读取已经确认过的结果。
CALIBRATION_DIR = Path("/home/gaoyuan/camera_hand_calibration/config")
DEFAULT_IP = "10.42.0.101"

# 工作台平面在机械臂基座坐标系中的 Z 高度，来自已完成的外部标定，单位 mm。
# 如果工作台高度改变，必须重新测量并更新这个值。
TABLE_Z_ARM_MM = 68.12483333

# CArm 的 tool=1 表示当前安装的夹爪 TCP。位置单位统一使用米。
TOOL = 1
GRASP_Z = 0.165

# 只读 IK 扫描 0°～165° 的 12 种试管方向后，这个点 12/12 可达。
# 当前现场只使用这一个固定高空转姿点，分别保存机械臂基座 X、Y，单位为米。
ROTATE_XY = (0.240, -0.010)

# 凹槽高空位使用上一张现场截图中的关节角，单位 rad。
GROOVE_ABOVE_JOINTS = np.array(
    [-0.986911, 1.985130, -0.395399, -0.727665, -1.554320, -0.039102]
)
GROOVE_ABOVE_REFERENCE_XYZ_M = np.array([0.223042, -0.207278, 0.230627])

# 凹槽释放位使用用户最新截图中的关节角，单位 rad。
GROOVE_RELEASE_JOINTS = np.array(
    [-0.988061, 2.029240, -0.378233, -0.729191, -1.570800, -0.081827]
)
GROOVE_RELEASE_REFERENCE_XYZ_M = np.array([0.222162, -0.206319, 0.216858])

# 四元数顺序固定为 [qx,qy,qz,qw]；它表示 yaw=0 时夹爪竖直向下。
DOWN_QUATERNION = (0.999575504, 0.008135427, 0.027844000, 0.002709061)

# 六个元素依次是 J1～J6，单位为弧度。
READY_JOINTS = np.array(
    [-0.001726, 1.751210, -0.626573, -0.000954, 0.446518, -0.000954]
)

# 彩色流分辨率和帧率必须与手眼标定时使用的相机模式一致。
IMAGE_WIDTH, IMAGE_HEIGHT, FPS = 1280, 720, 30

# 类别顺序就是 YOLO 类别 ID：0、1、2、3，不能随意交换。
MODEL_CLASSES = ("p-body", "p-cap", "y-body", "y-cap")

# 连续 10 帧中心抖动不超过 3 px、角度抖动不超过 3°，才允许按 C 锁定。
STABLE_FRAMES = 10
MAX_CENTER_JITTER_PX = 3.0
MAX_ANGLE_JITTER_DEG = 3.0
MAX_JOINT_STEP_DEG = 95.0

# 试管绕自身轴转多少度只影响机械臂腕部姿态。这里把尝试顺序直接列出来，
# 规划器从左到右逐个检查，找到第一条可达路径就停止。
TWIST_CANDIDATES = (
    0.0,
    15.0, -15.0,
    30.0, -30.0,
    45.0, -45.0,
    60.0, -60.0,
    75.0, -75.0,
    90.0, -90.0,
    105.0, -105.0,
    120.0, -120.0,
    135.0, -135.0,
    150.0, -150.0,
    165.0, -165.0,
    180.0,
)


# =============================================================================
# 2. 四元数：把平面 yaw 变成机械臂需要的姿态
# =============================================================================

def wrap_pi(angle: float) -> float:
    """把任意弧度角限制到 [-pi, pi)。

    ``%`` 是取模。角度加 pi 后对 2*pi 取模，再减 pi，就能把 370°、-350°
    等等价方向映射到统一范围，避免跨过 ±180° 时误判为巨大角度跳变。
    """

    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def normalize_quat(quaternion: Sequence[float]) -> tuple[float, ...]:
    """把四元数除以自身长度，使其成为单位四元数。"""

    # np.asarray 把列表/元组统一转换为浮点数组，便于做 shape 和有限值检查。
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,):
        raise ValueError("四元数必须正好包含四个数字")
    if not np.isfinite(q).all():
        raise ValueError("四元数不能包含 NaN 或无穷大")
    if np.linalg.norm(q) < 1e-12:
        raise ValueError("四元数长度不能为零")
    

    # 单位四元数必须满足 qx²+qy²+qz²+qw²=1。
    # np.linalg.norm(q) 用来计算四元数长度, 长度 = √(qx² + qy² + qz² + qw²)
    normalized = q / np.linalg.norm(q)  

    # normalized 原本是 NumPy 数组, tolist() 把它转换成普通Python列表
    qx, qy, qz, qw = normalized.tolist() 

    return qx, qy, qz, qw


def quat_multiply(left, right) -> tuple[float, ...]:
    # 这段函数的作用是：把两个旋转姿态合并成一个新的旋转姿态，先应用 right 旋转，再应用 left 旋转。
    # 例如：
    # right 表示“夹爪朝下”
    # left 表示“再绕基座 Z 轴旋转30°”
    # quat_multiply(left, right) 得到合并后的最终姿态 
    """两个 [qx,qy,qz,qw] 四元数做 Hamilton 积。

    旋转不能直接把四元数各分量相加；乘法表示旋转复合，而且先后顺序重要。
    """

    ax, ay, az, aw = normalize_quat(left)
    bx, by, bz, bw = normalize_quat(right)

    return normalize_quat(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    )

# down_pose_quat() 以“夹爪朝下姿态”为基础，再添加一个绕基座Z轴的平面旋转，得到用于对齐试管方向的最终四元数。
def down_pose_quat(yaw: float, down_quaternion=DOWN_QUATERNION) -> tuple[float, ...]:
    """保持夹爪向下，只绕基座 Z 轴旋转 yaw。"""
    # 表示函数返回一个浮点数元组：(qx, qy, qz, qw)

    # 旋转四元数使用半角：绕 Z 轴 yaw 对应 [0,0,sin(yaw/2),cos(yaw/2)]。
    half = wrap_pi(yaw) / 2.0
    yaw_quaternion = (0.0, 0.0, math.sin(half), math.cos(half))
    return quat_multiply(yaw_quaternion, down_quaternion)


# 把绕指定轴旋转指定角度转换成四元数, 先把旋转轴整理成单位方向，再利用半角的正弦和余弦生成四元数
def axis_angle_quat(axis, angle: float) -> tuple[float, ...]:

    axis = np.asarray(axis, dtype=float)

    axis_length = np.linalg.norm(axis)

    axis = axis / axis_length # 把旋转轴归一化为长度 1 的方向向量。

    half = angle / 2.0
    xyz = axis * math.sin(half)

    qx = float(xyz[0])
    qy = float(xyz[1])
    qz = float(xyz[2])
    qw = math.cos(half)

    quaternion = (qx, qy, qz, qw)
    return normalize_quat(quaternion)


def rotate_vector(quaternion, vector) -> np.ndarray:
    """用单位四元数旋转三维向量，不改变向量长度。"""

    qx, qy, qz, qw = normalize_quat(quaternion)
    q = np.array([qx, qy, qz])
    v = np.asarray(vector, dtype=float)
    # 这是 q * v * q⁻¹ 展开后的向量公式，可避免手工构造四元数 v。
    return (
        2.0 * np.dot(q, v) * q
        + (qw * qw - np.dot(q, q)) * v
        + 2.0 * qw * np.cross(q, v)
    )


def tilt_angles() -> list[float]:
    """简单版固定使用 45° 和 90° 两个目标姿态。"""

    # 之所以分成两段，是因为直接从水平姿态一次转到90°时，机械臂可能出现：
    # 逆运动学没有关节解；
    # 单次关节变化太大；
    # 接近关节限位；
    # 中间运动姿态不容易控制。
    # tilt_angles() 返回试管转正过程中的两个绝对目标角度：先倾斜到45°，再继续转到90°，最终让试管盖子朝上。

    return [45.0, 90.0]


def cap_up_orientation(grasp_q, tube_yaw: float, tilt_deg: float, twist_deg: float = 0.0):
    """计算指定倾斜角对应的四元数和盖子方向。

    ``direction`` 是基座坐标系中的水平 B→C；它与 +Z 的叉积给出翻转轴。
    ``twist_deg`` 是绕试管自身长轴转腕，不会改变盖子的朝向。
    """

    # grasp_q：夹爪已经对准并抓住水平试管时的四元数。
    # tube_yaw：试管 B→C 在机械臂基座 XY 平面中的方向，单位是弧度。
    # tilt_deg：准备把试管倾斜到多少度，例如45°或90°。
    # twist_deg：绕试管自身长轴转腕的总角度，默认0°。

    # 例如试管指向 +X：
    # direction = (1, 0, 0)
    # +Z        = (0, 0, 1)
    # rotation_axis = (0, -1, 0)
    # 也就是绕 -Y 轴翻转，才能把试管从 +X 方向逐渐转到 +Z。

    # 计算试管的水平 B→C 方向
    direction = np.array([math.cos(tube_yaw), math.sin(tube_yaw), 0.0]) 
    
    # 计算把试管向上翻转所需的旋转轴, [0, 0, 1] 是机械臂基座的 +Z 方向，也就是竖直向上。
    rotation_axis = np.cross(direction, [0.0, 0.0, 1.0]) 


    # 把翻转轴和翻转角转换成四元数, 绕 rotation_axis 旋转 tilt_deg
    tilt_q = axis_angle_quat(rotation_axis, math.radians(tilt_deg))

    # 计算倾斜后的试管盖子方向
    # 这里用刚才生成的 tilt_q 旋转试管原来的水平 B→C 方向。
    # 假设试管原来指向 +X：
    # direction = (1, 0, 0)
    # 倾斜45°后，大约是：
    # expected_direction = (0.707, 0, 0.707)
    # 表示试管一半朝前、一半朝上。
    # 倾斜90°后，大约是：
    # expected_direction = (0, 0, 1)
    # 表示 B→C 已经指向基座 +Z，也就是盖子朝上。
    expected_direction = rotate_vector(tilt_q, direction)

    # 夹爪倾斜后的目标姿态, 在当前抓取姿态的基础上，再执行试管向上翻转
    target_q = quat_multiply(tilt_q, grasp_q)

    # 非零数字在 Python 中视为 True，因此 twist_deg=0 时跳过这段计算。
    if twist_deg:
        twist_q = axis_angle_quat(
            expected_direction,
            math.radians(twist_deg) * tilt_deg / 90.0, # 把绕指定轴旋转指定角度转换成四元数
        )

        target_q = quat_multiply(twist_q, target_q)

    return {
        "tilt_deg": tilt_deg,
        "quaternion": target_q,
        "cap_direction": expected_direction,
    }


def cap_up_orientations(grasp_q, tube_yaw: float, twist_deg: float = 0.0):
    """返回真机依次执行的 45° 和 90° 转姿目标。"""

    steps = []
    for tilt_deg in tilt_angles():
        step = cap_up_orientation(grasp_q, tube_yaw, tilt_deg, twist_deg)
        steps.append(step)
    return steps


# =============================================================================
# 3. RealSense + YOLO：得到图像中的中心、底部 B 和盖子 C
# =============================================================================

def latest_model() -> str:
    """在 runs 目录中查找修改时间最新的 best.pt。"""

    latest_path = None
    latest_time = -1.0

    # rglob 会递归查找 runs 下所有名为 best.pt 的路径。
    for path in (ROOT / "runs").rglob("best.pt"):
        if not path.is_file():
            continue
        modified_time = path.stat().st_mtime

        if latest_path is None or modified_time > latest_time:
            latest_path = path
            latest_time = modified_time

    if latest_path is None:
        raise FileNotFoundError("runs 下没有 best.pt，请用 --model 指定权重")
    return str(latest_path)


def ordered_model_names(names: Any) -> tuple[str, ...]:
    """把 Ultralytics 的类别名统一整理成按 ID 排序的元组。"""

    # 先准备一个空列表，用来依次保存整理后的类别名称。
    ordered_names = []

    # 有些模型返回 {0: "p-body", ...}，需要先按照类别 ID 排序。
    if isinstance(names, dict): # 判断是不是字典，isinstance() 用来检查一个变量是不是指定类型。
        for class_id in sorted(names):
            class_name = names[class_id]
            ordered_names.append(str(class_name))
    else:
        # 如果模型已经返回列表或元组，就保持它原来的类别顺序。
        for class_name in names:
            ordered_names.append(str(class_name))

    # 最后统一转换成元组，便于和固定的 MODEL_CLASSES 比较。
    return tuple(ordered_names)


def load_matrix(filename: str, expected_shape) -> np.ndarray:
    """读取一个逗号分隔的矩阵文件，并检查尺寸和数值。"""

    value = np.loadtxt(CALIBRATION_DIR / filename, delimiter=",")
    if value.shape != expected_shape:
        raise ValueError(f"{filename} 应为 {expected_shape}，实际为 {value.shape}")
    if not np.isfinite(value).all(): # np.isfinite(value) 检查矩阵中的每个数字是不是有限数字
        raise ValueError(f"{filename} 包含 NaN 或无穷大")
    return value


def load_calibration() -> dict[str, Any]:
    """读取相机内参和“相机坐标 → 机械臂基座坐标”矩阵。

    外部工程的文件名是 ``T_arm2cam.txt``；根据现有矩阵组合关系，它实际
    用于把相机坐标变换到机械臂基座坐标。矩阵平移单位是毫米。
    """

    # with 会在代码块结束时自动关闭文件，即使读取过程中发生异常。
    with (CALIBRATION_DIR / "camera.yaml").open(encoding="utf-8") as file:

        camera = yaml.safe_load(file) # 把 YAML 文件转换成Python字典

    # K 是 3×3 相机内参；T_arm2cam 是 4×4 齐次坐标变换矩阵。
    intrinsic = load_matrix("intrinsic.txt", (3, 3))
    arm_from_camera = load_matrix("T_arm2cam.txt", (4, 4)) # 相机坐标系 → 机械臂基座坐标系
 
    # 字典用有意义的键名同时返回多个相关结果。
    return {
        "intrinsic": intrinsic,
        "arm_from_camera": arm_from_camera,
        "width": int(camera["img_width"]),
        "height": int(camera["img_height"]),
    }


def pixel_to_arm(pixel, calibration) -> np.ndarray:
    """把图像像素变成机械臂基座坐标系中的工作台点，结果单位为米。

    计算顺序是：像素 -> 相机射线 -> 机械臂射线 -> 固定工作台平面交点。
    单个 RGB 像素本身没有深度；这里利用工作台固定高度求出三维位置。
    """

    # 序列解包：pixel 的两个元素分别赋给水平像素 u 和垂直像素 v。
    u, v = np.asarray(pixel, dtype=float)

    # 不显式计算 K 的逆，而是解线性方程 K·ray=[u,v,1]，数值上更稳健。
    ray_camera = np.linalg.solve(calibration["intrinsic"], [u, v, 1.0])

    # T_arm2cam 的左上 3×3 是旋转，右上 3×1 是相机原点的机械臂坐标。
    arm_from_camera = calibration["arm_from_camera"]

    ray_origin_arm = arm_from_camera[:3, 3]

    ray_direction_arm = arm_from_camera[:3, :3] @ ray_camera

    if abs(float(ray_direction_arm[2])) < 1e-12:
        raise ValueError("像素射线与工作台平面平行")

    # 射线方程 p=origin+scale·direction；令 p_z=TABLE_Z_ARM_MM 求 scale。
    scale = (TABLE_Z_ARM_MM - ray_origin_arm[2]) / ray_direction_arm[2]

    if not math.isfinite(float(scale)):
        raise ValueError("像素射线交点计算结果不是有限数字")
    if scale <= 0:
        raise ValueError("像素射线不能与工作台正向相交")
    arm_mm = ray_origin_arm + scale * ray_direction_arm
    # 消除浮点误差，明确结果就在固定工作台平面上。
    arm_mm[2] = TABLE_Z_ARM_MM
    # 标定矩阵用毫米；CArm Pose 的位置使用米。
    return arm_mm[:3] / 1000.0


class RealSenseColor:
    """RealSense 彩色相机的小型包装类，只打开 1280×720 彩色流。

    类把 ``pipeline`` 和实时内参保存在同一个对象中；``self`` 代表当前对象。
    """

    def __init__(self, serial: str | None):
        # 延迟导入：只有真正打开相机时才要求安装 pyrealsense2。
        """初始化当前对象，并保存后续操作需要的状态。"""

        import pyrealsense2 as rs

        self.pipeline = rs.pipeline()
        config = rs.config()
        # serial 的类型是“字符串或 None”；没有指定时使用系统找到的相机。
        if serial:
            config.enable_device(serial)
        config.enable_stream(
            rs.stream.color,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            rs.format.bgr8,
            FPS,
        )
        profile = self.pipeline.start(config)
        device = profile.get_device() # 获取相机设备
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = stream.get_intrinsics()

        # 保存 fx、fy、cx、cy，稍后与手眼标定时的内参进行核对。
        self.intrinsics = np.array(
            [intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy]
        )
        actual_serial = device.get_info(rs.camera_info.serial_number) # 读取实际相机序列号
        print(f"相机：{actual_serial}，{stream.width()}x{stream.height()}@{stream.fps()}")

    def read(self):
        """最多等待 5 秒并返回一帧 BGR 图像；失败时返回 None。"""

        frame = self.pipeline.wait_for_frames(5000).get_color_frame() # 程序最多等待5秒获取新的相机数据
        if frame is None:
            return None
        image = np.asanyarray(frame.get_data()) # 从数据中取出彩色图像帧
        return image

    def close(self):
        """停止相机数据流，释放 USB/设备资源。"""

        self.pipeline.stop()


def largest_component(mask: np.ndarray) -> np.ndarray:
    """只保留二值掩膜中面积最大的连通区域，去掉零碎噪点。"""

    # labels 给每块连通区域编号；stats 保存每块区域的矩形和面积。
    # 编号 0 是背景，所以实际物体从编号 1 开始。
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8)) # 会把彼此连接的白色像素划分成不同区域
    if count < 2:
        raise ValueError("掩膜为空")
    
    # argmax 返回面积最大元素在 stats[1:] 中的位置，因此最后要加 1。
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    # 比较会产生布尔数组：最大区域为 True，其余区域为 False。
    return labels == index


def mask_points(masks, index: int, width: int, height: int) -> np.ndarray:
    """把一个实例掩膜转换成 N×2 的 [x,y] 像素点数组。"""

    # YOLO 掩膜可能不是原图尺寸，先缩放，再用 0.5 阈值二值化。
    mask = cv2.resize(masks[index], (width, height)) > 0.5
    # np.nonzero 对图像返回的顺序是 y、x，所以这里明确写成 ys、xs。
    ys, xs = np.nonzero(largest_component(mask))

    return np.column_stack((xs, ys)).astype(float)


def yellow_geometry(result, image_shape):
    """从 y-body/y-cap 掩膜计算试管中心、底部 B 和盖子端 C。

    B→C 是有方向的长轴：B 指试管无盖端，C 指黄色盖子端。
    """

    if result.boxes is None or result.masks is None:
        return None
    # 推理张量可能在 GPU 上：detach -> cpu -> numpy 后才能交给 NumPy/OpenCV。
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)

    # MODEL_CLASSES 中 y-body 的 ID 是 2，y-cap 的 ID 是 3。
    # flatnonzero 返回满足条件的实例下标。
    bodies = np.flatnonzero(class_ids == 2)
    caps = np.flatnonzero(class_ids == 3)

    # 简单版只处理画面里恰好一个黄色管身和一个黄色管盖的情况。
    if len(bodies) != 1 or len(caps) != 1:
        return None

    masks = result.masks.data.detach().cpu().numpy() # 取得所有实例掩膜
    height, width = image_shape
 
    body = mask_points(masks, int(bodies[0]), width, height) # 把管身和管盖掩膜转换成像素点
    cap = mask_points(masks, int(caps[0]), width, height)

    if len(body) < 30 or len(cap) < 5:
        return None

    # 管身所有像素的均值作为中心。axis=0 表示分别对 x 列和 y 列求均值。
    center = np.mean(body, axis=0)

    # PCA 主轴：先求 2×2 协方差矩阵，再取最大特征值对应的特征向量。
    covariance = np.cov((body - center).T)
    values, vectors = np.linalg.eigh(covariance)

    axis = vectors[:, int(np.argmax(values))]

    # @ 在这里是“每个中心化像素点与主轴做点积”，得到沿长轴的一维投影。
    projections = (body - center) @ axis

    # 用 1%/99% 分位数代替绝对最小/最大，降低少量边缘噪点的影响。
    end_1 = center + np.percentile(projections, 1) * axis
    end_2 = center + np.percentile(projections, 99) * axis
    if np.linalg.norm(end_2 - end_1) < 30:
        return None

    # 哪一个管身端点更靠近管盖中心，哪一个就是有盖端 C；另一端就是 B。
    cap_center = np.mean(cap, axis=0)
    if np.linalg.norm(cap_center - end_1) < np.linalg.norm(cap_center - end_2):
        bottom, cap_end = end_2, end_1
    else:
        bottom, cap_end = end_1, end_2
    return center, bottom, cap_end


def stable_geometry(history):
    """检查最近若干帧的中心和方向是否足够稳定，并返回中位结果。"""

    if len(history) < STABLE_FRAMES:
        return None
    centers_list = []
    bottoms_list = []
    caps_list = []
    for center_px, bottom_px, cap_px in history:
        centers_list.append(center_px)
        bottoms_list.append(bottom_px)
        caps_list.append(cap_px)

    centers = np.asarray(centers_list)
    bottoms = np.asarray(bottoms_list)
    caps = np.asarray(caps_list)
    # 中位数比均值更不容易被某一帧的异常检测拉偏。
    center = np.median(centers, axis=0)
    # 每帧中心到中位中心的欧氏距离都必须不超过阈值。
    if np.max(np.linalg.norm(centers - center, axis=1)) > MAX_CENTER_JITTER_PX:
        return None
    # atan2(dy,dx) 计算每帧 B→C 相对图像 x 正方向的有向角。
    angles = np.arctan2((caps - bottoms)[:, 1], (caps - bottoms)[:, 0])
    # 角度不能直接做普通平均。例如 +179° 和 -179° 的平均方向应接近 180°，
    # 而不是 0°。先平均 sin/cos，再用 atan2 恢复角度，称为圆周平均。
    mean = math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
    angle_errors = []
    for angle in angles:
        angle_errors.append(abs(wrap_pi(angle - mean)))
    max_angle_error_deg = math.degrees(max(angle_errors))
    if max_angle_error_deg > MAX_ANGLE_JITTER_DEG:
        return None
    return center, np.median(bottoms, axis=0), np.median(caps, axis=0)


def make_solution(geometry, calibration, yaw_offset_deg: float) -> dict[str, Any]:
    """把稳定的像素几何转换为机械臂平面位置和两个等价抓取 yaw。"""

    center, bottom, cap = geometry
    # 中心、B、C 分别投影到同一个机械臂基座坐标系。
    center_arm = pixel_to_arm(center, calibration)
    bottom_arm = pixel_to_arm(bottom, calibration)
    cap_arm = pixel_to_arm(cap, calibration)

    direction = cap_arm - bottom_arm
    # atan2(y,x) 得到机械臂 XY 平面内的 B→C 方向。
    tube_yaw = math.atan2(float(direction[1]), float(direction[0]))
    yaw_1 = wrap_pi(tube_yaw + math.radians(yaw_offset_deg))

    # 对称夹爪绕 Z 转 180° 仍可夹住同一条轴，所以保留两个 IK 候选。
    return {
        "center_arm": center_arm,
        "tube_yaw": tube_yaw,
        "candidate_yaws": (yaw_1, wrap_pi(yaw_1 + math.pi)),
    }


def draw_live_preview(result, geometry, solution) -> np.ndarray:
    """绘制简洁实时画面：只显示掩膜、B/C 点和机械臂基座 yaw。

    ``result.plot`` 关闭检测框、类别文字和置信度，只保留实例分割掩膜。
    B 是试管无盖端，C 是靠近黄色盖子的管身端点；稳定后显示的 yaw 与
    ``make_solution`` 交给机械臂规划的 ``tube_yaw`` 完全相同。
    """

    shown = result.plot(
        masks=True,
        boxes=False,
        labels=False,
        conf=False,
    )

    image_angle_text = "IMAGE B->C ANGLE: waiting"

    if geometry is not None:
        # OpenCV 绘图坐标必须是整数元组。
        center = tuple(np.round(geometry[0]).astype(int))
        bottom = tuple(np.round(geometry[1]).astype(int))
        cap = tuple(np.round(geometry[2]).astype(int))

        # 黄色线表示有方向的 B→C 长轴；蓝色 B 是无盖端，红色 C 是盖子端。
        cv2.line(shown, bottom, cap, (0, 255, 255), 3)
        cv2.circle(shown, bottom, 7, (255, 0, 0), -1)
        cv2.circle(shown, cap, 7, (0, 0, 255), -1)
        cv2.circle(shown, center, 5, (0, 255, 0), -1)

        # 图像坐标是 +u 向右、+v 向下；因此该角度只用于检查图像几何，
        # 不能直接当成机械臂基座 yaw。
        delta_u = float(geometry[2][0] - geometry[1][0])
        delta_v = float(geometry[2][1] - geometry[1][1])
        image_angle_deg = math.degrees(math.atan2(delta_v, delta_u))
        image_angle_text = f"IMAGE B->C ANGLE: {image_angle_deg:.1f} deg"

        cv2.putText(
            shown,
            "B",
            (bottom[0] + 9, bottom[1] - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            shown,
            "C",
            (cap[0] + 9, cap[1] - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if solution is not None:
        status_text = "STABLE - C: lock"
        status_color = (0, 255, 0)
        yaw_deg = math.degrees(solution["tube_yaw"])
        yaw_text = f"BASE B->C YAW: {yaw_deg:.1f} deg"

    else:
        status_text = "Waiting for y-body/y-cap"
        status_color = (0, 165, 255)
        yaw_text = "BASE B->C YAW: waiting"

    cv2.putText(
        shown,
        status_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        shown,
        image_angle_text,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        shown,
        yaw_text,
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
        cv2.LINE_AA,
    )

    return shown


def detect_and_lock(args, calibration) -> dict[str, Any]:
    """实时显示分割结果；中心和方向稳定后按 C 锁定目标。"""

    # 延迟导入让 ``python script.py --help`` 不必先加载较重的深度学习库。
    from ultralytics import YOLO

    if args.model:
        model_path = args.model
    else:
        model_path = latest_model()
    model = YOLO(model_path)

    # 同时核对任务类型与四类名称，避免拿错检测模型或类别顺序错位。
    if str(model.task) != "segment":
        raise ValueError("模型任务必须是实例分割 segment")
    
    model_names = ordered_model_names(model.names)

    if model_names != MODEL_CLASSES:
        raise ValueError(f"模型类别应为 {MODEL_CLASSES}，实际为 {model_names}")
    
    print(f"模型：{model_path}")

    camera = RealSenseColor(args.serial)
    saved = calibration["intrinsic"]
    expected = [saved[0, 0], saved[1, 1], saved[0, 2], saved[1, 2]]

    # 实时内参与标定内参最大相差不得超过 1 像素。
    error = float(np.max(np.abs(camera.intrinsics - expected)))

    if error > 1.0:
        camera.close()
        raise ValueError(f"实时内参与标定不一致：{error:.3f} px")

    # maxlen 满后会自动丢弃最老的一帧，始终只保留最近 STABLE_FRAMES 帧。
    history = deque(maxlen=STABLE_FRAMES)

    window = "14 - simple yellow tube cap up"

    print("目标稳定后按 C；按 Q 或 Esc 退出。")
    try:
        while True:  # 无限循环，直到用户锁定目标或按 Q/Esc 退出。
            frame = camera.read() # frame 是BGR图像，形状大致为，(720, 1280, 3)
            if frame is None:
                continue

            result = model.predict(
                frame,
                # imgsz 是网络输入大小；conf/iou 是置信度和 NMS 阈值。
                imgsz=1024,
                conf=0.90,
                iou=0.70,
                device="0",
                # 项目视觉脚本统一固定使用 FP32。
                half=False,
                retina_masks=True,
                verbose=False,
            )[0]

            geometry = yellow_geometry(result, frame.shape[:2])

            # 当前帧有效就加入历史；无效则清空，要求稳定帧必须连续。
            if geometry is not None:
                history.append(geometry)
            else:
                history.clear() 

            stable = stable_geometry(history)
            
            if stable is not None:
                solution = make_solution(stable, calibration, args.yaw_offset_deg)
            else:
                solution = None

            shown = draw_live_preview(result, geometry, solution) # 绘制实时预览

            cv2.imshow(window, shown)
            # waitKey 返回平台相关整数；& 0xFF 只保留最低 8 位按键码。
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                raise KeyboardInterrupt
            if key in (ord("c"), ord("C")) and solution:
                return solution
    finally:
        # 无论正常 return、用户取消还是发生异常，都关闭相机和窗口。
        camera.close()
        cv2.destroyAllWindows()


# =============================================================================
# 4. CArm：只保留本 Demo 使用的几个命令
# =============================================================================


class SimpleCArm:
    """连接、IK/FK、移动和夹爪的最小包装。

    这个类没有重新实现机械臂算法，只是把第三方 ``carm`` SDK 的返回格式、
    单位检查和常用参数集中到一个地方，主流程因此更容易阅读。
    """

    def __init__(self, ip: str):
        # 同样采用延迟导入：纯视觉预览时完全不会加载或连接 CArm。
        """初始化当前对象，并保存后续操作需要的状态。"""

        from carm import Carm

        # Carm(...) 自己会启动 WebSocket 连接线程。这里等待它完成，不能马上再
        # 调用 connect()，否则会先断开尚在建立的连接并创建第二个线程。
        self.arm = Carm(addr=ip, arm_index=0)
        # monotonic() 是只向前走的计时钟，不会受电脑系统时间调整影响。
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not self.arm.is_connected():
            time.sleep(0.05)
        if not self.arm.is_connected():
            self.arm.disconnect()
            raise RuntimeError(f"无法连接 CArm：{ip}")

    def close(self):
        """关闭 WebSocket 连接。"""

        self.arm.disconnect()

    def check_command(self, name: str, response):
        """按照当前 CArm SDK 的两种返回格式检查命令是否成功。"""

        # set_ready() 成功时直接返回 True。
        if response is True:
            return

        # 其余控制命令成功时返回包含 Task_Recieve 的字典。
        if isinstance(response, dict):
            if response.get("recv") == "Task_Recieve":
                return

        raise RuntimeError(f"{name} 失败：{response}")

    def ready(self):
        """使能机械臂并设置控制模式、速度、工具和碰撞配置。"""

        response = self.arm.set_ready()
        self.check_command("set_ready", response)

        response = self.arm.set_control_mode(1)
        self.check_command("set_control_mode", response)

        # 当前现场值是 set_speed_level(3, 80)。这是设备速度配置，不是 m/s。
        response = self.arm.set_speed_level(1.5, 80)
        self.check_command("set_speed_level", response)

        # tool=1 告诉控制器使用已安装夹爪对应的 TCP 标定。
        response = self.arm.set_tool_index(TOOL)
        self.check_command("set_tool_index", response)

        response = self.arm.set_collision_config(True, 2)
        self.check_command("set_collision_config", response)

    def fk(self, joints) -> list[float]:
        """正运动学 FK：由六个关节角计算 TCP 的七维 Pose。"""

        # Pose=[x,y,z,qx,qy,qz,qw]，其中位置单位是米。
        pose = np.asarray(self.arm.forward_kine(list(joints), tool=TOOL), dtype=float)

        if pose.shape != (7,):
            raise RuntimeError(f"FK 应返回 7 个数字，实际形状为 {pose.shape}")
        
        if not np.isfinite(pose).all():
            raise RuntimeError("FK Pose 包含 NaN 或无穷大")
        
        pose[3:] = normalize_quat(pose[3:]) # 取出第4个元素到最后： [qx, qy, qz, qw]，归一化后再写回 Pose 的最后四个位置。

        return pose.tolist()

    def joint_limits(self):
        """等待并返回 SDK 给出的六关节下限和上限。"""

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            limits = getattr(self.arm, "limit", {})
            lower = np.asarray(limits.get("limit_lower", []), dtype=float)
            upper = np.asarray(limits.get("limit_upper", []), dtype=float)
            if lower.shape == (6,) and upper.shape == (6,):
                return lower, upper
            time.sleep(0.05)
        raise RuntimeError("读取关节限位超时")

    def current_joints(self) -> np.ndarray:
        """连接成功后，等待第一帧有效的六关节状态。"""

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            joints = np.asarray(self.arm.joint_pos, dtype=float)
            if joints.shape == (6,) and np.isfinite(joints).all():
                return joints
            time.sleep(0.05)
        raise RuntimeError("已连接 CArm，但 3 秒内没有收到有效六关节状态")

    def ik(self, pose, seed, max_step_deg=MAX_JOINT_STEP_DEG) -> np.ndarray:
        """逆运动学 IK：由目标 Pose 求一组六关节角，并做目标一致性检查。

        ``seed`` 是求解初值。相同末端 Pose 可能有多组关节解，使用上一段关节角
        作为 seed，通常能得到更连续的路径。
        """

        response = self.arm.inverse_kine(list(pose), list(seed), tool=TOOL)

        try:
            joints = np.asarray(response["data"]["joint1"], dtype=float)

        # 元组形式 except (...) 表示下面三类解析错误都按同一种方式处理。
        except (KeyError, TypeError, ValueError) as exc:
            # ``raise ... from exc`` 保留底层原因，方便调试异常链。
            raise RuntimeError("IK 没有返回关节解") from exc
        
        if joints.shape != (6,):
            raise RuntimeError(f"IK 应返回 6 个关节角，实际形状为 {joints.shape}")
        
        if not np.isfinite(joints).all():
            raise RuntimeError("IK 关节解包含 NaN 或无穷大")
        
        lower, upper = self.joint_limits()
        # 每个关节与物理限位至少保留 2° 余量；计算时转换成弧度。
        margin = math.radians(2.0)

        if np.any(joints < lower + margin) or np.any(joints > upper - margin):
            raise RuntimeError("IK 关节解太靠近限位")
        
        # 六个关节分别求变化量，取其中最大的一个作为该段路径的变化指标。
        step = math.degrees(float(np.max(np.abs(joints - np.asarray(seed)))))

        if step > max_step_deg:
            raise RuntimeError(f"单段关节变化 {step:.1f}° > {max_step_deg:.1f}°")

        # 用 FK 把 IK 解算回 Pose，验证它确实接近请求的目标，而非只相信 SDK 响应。
        actual = np.asarray(self.fk(joints)) # actual 是将IK关节解重新送入FK得到的Pose
        target = np.asarray(pose, dtype=float) # target 是要求的目标Pose

        if np.linalg.norm(actual[:3] - target[:3]) > 0.002:
            raise RuntimeError("IK/FK 位置误差超过 2 mm")
        
        # 单位四元数 q 和 -q 表示同一旋转，所以点积取绝对值。
        # 两个单位四元数的姿态误差角为 2*acos(|q1·q2|)。
        quaternion_dot = abs(float(np.dot(actual[3:], target[3:])))
        angle_error = 2.0 * math.acos(float(np.clip(quaternion_dot, 0.0, 1.0))) # np.clip() 把点积限制在 0～1，避免浮点误差导致 acos() 输入略大于1

        if math.degrees(angle_error) > 1.0:
            raise RuntimeError("IK/FK 姿态误差超过 1°")
        
        return joints

    def move_joints(self, joints):
        """使用关节空间运动到目标六关节角，并同步等待命令完成。"""

        response = self.arm.move_joint(list(joints), is_sync=True)
        self.check_command("move_joint", response)

    def move_line(self, pose):
        """让 TCP 沿笛卡尔直线路径移动到目标 Pose。"""

        response = self.arm.move_line_pose(list(pose), is_sync=True, tool=TOOL)
        self.check_command("move_line_pose", response)

    def open_gripper(self):
        """把夹爪目标开口设为 60 mm，第二个参数沿用现场力/速度设置。"""

        response = self.arm.set_gripper(0.060, 10.0)
        self.check_command("open_gripper", response)

    def close_gripper(self):
        """把夹爪目标开口设为 0 mm。"""

        response = self.arm.set_gripper(0.000, 10.0)
        self.check_command("close_gripper", response)


# =============================================================================
# 5. 路径搜索：抓取 twist、盖朝上和固定凹槽关节位
# =============================================================================

def pose(x, y, z, quaternion) -> list[float]:
    """组合成 CArm 使用的 [x,y,z,qx,qy,qz,qw] Pose。"""

    qx, qy, qz, qw = normalize_quat(quaternion)
    return [float(x), float(y), float(z), qx, qy, qz, qw] # 这个函数只是把位置和姿态装进一个列表


def build_cap_up_plan(robot: SimpleCArm, solution, rotate_z: float):
    """只读搜索第一条完整可达路径；找不到就停止，不移动机械臂。

    搜索顺序：高度 -> 抓取 yaw -> 盖朝上 twist。
    抓取和转姿逐段通过 IK 后，只核对固定高空位和释放位，不搜索凹槽 IK。
    """

    # center_arm 有 x、y、z；下划线表示这里不使用检测得到的 z。
    x, y, _ = solution["center_arm"]

    # 准备位的朝向和 XY 来自固定关节位，随后仅把高度替换为候选高度。
    ready_reference = robot.fk(READY_JOINTS)

    # 每次检查失败都保存“失败阶段 + 原因”，最后按阶段汇总。
    failures = []

    # 即使用户给的 rotate_z 不可达，还会自动尝试高 25 mm 和 50 mm。
    for height in (rotate_z, rotate_z + 0.025, rotate_z + 0.050):
        ready_pose = ready_reference.copy()
        ready_pose[2] = height

        try:
            ready_joints = robot.ik(ready_pose, READY_JOINTS)

        except RuntimeError as exc: # 如果不可达，保存错误，跳到下一个高度
            failures.append(("高空准备位", str(exc)))
            continue

        yaw_index = 0

        for yaw in solution["candidate_yaws"]:
            yaw_index += 1
            grasp_q = down_pose_quat(yaw) # 夹爪保持垂直向下，并在工作台平面内对齐试管方向
            above_pose = pose(x, y, height, grasp_q)
            grasp_pose = pose(x, y, GRASP_Z, grasp_q)

            stage = "试管上方"
            try:
                above_joints = robot.ik(above_pose, ready_joints, 95.0)
                stage = "抓取下降位"
                robot.ik(grasp_pose, above_joints)

            except RuntimeError as exc:
                failures.append((stage, str(exc)))
                continue

            rotate_x, rotate_y = ROTATE_XY
            transfer_pose = pose(rotate_x, rotate_y, height, grasp_q)

            try:
                transfer_joints = robot.ik(transfer_pose, above_joints) # 它保持当前抓取姿态，只把夹爪水平移动到高空转姿位置
            except RuntimeError as exc:
                failures.append(("高空转姿点", str(exc)))
                continue

            # twist 只改变腕部绕试管轴的姿态，帮助避开关节限位或无解区域。
            for twist in TWIST_CANDIDATES:
                previous_joints = transfer_joints
                tilt_steps = []

                stage = "盖朝上转姿"
                try:
                    for item in cap_up_orientations(
                        grasp_q, solution["tube_yaw"], twist
                    ):
                        stage = f"{item['tilt_deg']:.0f}°盖朝上转姿"
                        target_pose = pose(
                            rotate_x, rotate_y, height, item["quaternion"]
                        )
                        # 45° 解作为 90° 的 seed，使两段 IK 和实际运动保持一致。
                        joints = robot.ik(target_pose, previous_joints)

                        # 真机执行只需要显示倾斜角并发送关节角，不保存其他重复数据。
                        tilt_steps.append(
                            {
                                "tilt_deg": item["tilt_deg"],
                                "joints": joints,
                            }
                        )

                        previous_joints = joints # 让下一段IK从当前姿态继续计算

                except RuntimeError as exc:
                    failures.append((stage, str(exc)))
                    continue

                # 90° 转姿完成后，依次检查两个固定凹槽关节位。
                cap_up_joints = tilt_steps[-1]["joints"]

                groove_above_joints = GROOVE_ABOVE_JOINTS.copy()
                groove_above_pose = robot.fk(groove_above_joints)
                above_position_error_m = np.linalg.norm(
                    np.asarray(groove_above_pose[:3])
                    - GROOVE_ABOVE_REFERENCE_XYZ_M
                )
                if above_position_error_m > 0.002:
                    failures.append(
                        (
                            "固定凹槽高空位",
                            "FK 与截图位置相差 "
                            f"{above_position_error_m * 1000.0:.1f} mm",
                        )
                    )
                    continue

                above_step_deg = math.degrees(
                    float(
                        np.max(np.abs(groove_above_joints - cap_up_joints))
                    )
                )
                if above_step_deg > MAX_JOINT_STEP_DEG:
                    failures.append(
                        (
                            "固定凹槽高空位",
                            f"单段关节变化 {above_step_deg:.1f}° > "
                            f"{MAX_JOINT_STEP_DEG:.1f}°",
                        )
                    )
                    continue

                groove_release_joints = GROOVE_RELEASE_JOINTS.copy()
                groove_release_pose = robot.fk(groove_release_joints)
                release_position_error_m = np.linalg.norm(
                    np.asarray(groove_release_pose[:3])
                    - GROOVE_RELEASE_REFERENCE_XYZ_M
                )
                if release_position_error_m > 0.002:
                    failures.append(
                        (
                            "固定凹槽释放位",
                            "FK 与截图位置相差 "
                            f"{release_position_error_m * 1000.0:.1f} mm",
                        )
                    )
                    continue

                release_step_deg = math.degrees(
                    float(
                        np.max(
                            np.abs(groove_release_joints - groove_above_joints)
                        )
                    )
                )
                if release_step_deg > MAX_JOINT_STEP_DEG:
                    failures.append(
                        (
                            "固定凹槽释放位",
                            f"单段关节变化 {release_step_deg:.1f}° > "
                            f"{MAX_JOINT_STEP_DEG:.1f}°",
                        )
                    )
                    continue

                return {
                    "yaw_index": yaw_index,
                    "grasp_yaw": yaw,
                    "twist_deg": twist,
                    "rotate_z": height,
                    "ready_joints": ready_joints,
                    "above_pose": above_pose,
                    "above_joints": above_joints,
                    "grasp_pose": grasp_pose,
                    "transfer_joints": transfer_joints,
                    "vertical_steps": tilt_steps,
                    "groove_above_pose": groove_above_pose,
                    "groove_above_joints": groove_above_joints,
                    "groove_release_pose": groove_release_pose,
                    "groove_release_joints": groove_release_joints,
                }

    if not failures:
        raise RuntimeError("没有找到简单可达路径：没有 IK 解")

    # 同一阶段可能在不同高度、yaw、twist 下失败多次。这里合并相同阶段，
    # 只打印失败次数和该阶段最后一次的 SDK 原因，避免终端输出几百行。
    failure_summary = {}
    for stage, reason in failures:
        if stage not in failure_summary:
            failure_summary[stage] = {"count": 0, "last_reason": ""}
        failure_summary[stage]["count"] += 1
        failure_summary[stage]["last_reason"] = reason

    lines = ["没有找到简单可达路径，各阶段 IK 失败汇总："]
    for stage, information in failure_summary.items():
        lines.append(
            f"  {stage}：失败 {information['count']} 次；"
            f"最后原因：{information['last_reason']}"
        )
    raise RuntimeError("\n".join(lines))


# =============================================================================
# 6. 真正执行动作
# =============================================================================

def execute_plan(robot: SimpleCArm, solution, plan):
    """用户确认后，执行抓取、盖朝上，并到固定凹槽释放位。"""

    x, y, _ = solution["center_arm"]
    # 相邻字符串字面量会自动拼接；f 前缀允许把变量和格式写进字符串。
    confirmation = (
        f"GRASP_CAP_UP_SIMPLE tool=1 x={x:.3f} y={y:.3f} "
        f"z={GRASP_Z:.3f} yaw={math.degrees(plan['grasp_yaw']):.1f} "
        f"groove=({plan['groove_release_pose'][0]:.3f},"
        f"{plan['groove_release_pose'][1]:.3f},"
        f"{plan['groove_release_pose'][2]:.3f})"
    )
    print("请清空完整运动区域，并准备好急停。")

    # input() 读取一行；strip() 去掉首尾空格和换行，必须与计划文字完全相同。
    if input(f"请输入完整确认文字：\n{confirmation}\n> ").strip() != confirmation:
        print("确认文字不匹配，没有运动。")
        return

    robot.ready() # 先使能并设置模式、速度、工具、碰撞配置，再移动到准备位

    print("1/10：到高空准备位")
    robot.move_joints(plan["ready_joints"])

    print("2/10：到试管上方并张开夹爪")
    robot.move_joints(plan["above_joints"]) # 移动到试管中心正上方并张开夹爪。
    robot.open_gripper()


    print("3/10：直线下降")
    robot.move_line(plan["grasp_pose"]) # 保持当前抓取yaw，沿笛卡尔直线下降。


    print("4/10：等待一段时间关闭爪子")
    time.sleep(0.5)
    robot.close_gripper()


    print("5/10：直线抬高")
    robot.move_line(plan["above_pose"])


    print("6/10：移动到高空转姿点")
    robot.move_joints(plan["transfer_joints"])


    print("7/10：分 45°、90° 两段把盖子转向 +Z")  # 最终让B→C指向基座 +Z。
    # for 循环依次执行两个已经通过 IK/FK 检查的关节位。
    for step in plan["vertical_steps"]:
        print(f"  倾斜 {step['tilt_deg']:.0f}°")
        robot.move_joints(step["joints"])
    print("8/10：盖子已经朝上")

    print("9/10：移动到固定凹槽高空位")
    robot.move_joints(plan["groove_above_joints"])

    print("10/10：移动到新的固定凹槽释放位并暂停")
    robot.move_joints(plan["groove_release_joints"])
    print("已经到达固定凹槽释放位，程序暂停在这里。")


# =============================================================================
# 7. 命令行和主流程——初学者建议先读这里
# =============================================================================

def parse_args():
    """定义命令行选项并把用户输入解析成 args 对象。"""

    parser = argparse.ArgumentParser(description="试管抓取Demo")
    parser.add_argument("--model", help="YOLO Seg 权重；默认最新 best.pt")
    parser.add_argument("--serial", help="可选 RealSense 序列号")

    # type=float 负责把命令行字符串转成浮点数；default 是没有填写时的值。
    parser.add_argument("--yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--rotate-z", type=float, default=0.300)  # 高空转姿高度

    parser.add_argument("--ip", default=DEFAULT_IP)

    parser.add_argument("--check-ik", action="store_true", help="只检查 IK，不运动")
    parser.add_argument("--execute", action="store_true", help="确认后真实执行")
    
    return parser.parse_args()

def main() -> int:
    
    args = parse_args() # 读取参数
    
    calibration = load_calibration() # 首先读取标定，然后打开相机让用户锁定黄色试管
    print("标定读取成功：intrinsic.txt + T_arm2cam.txt")
    
    solution = detect_and_lock(args, calibration) # 相机检测并锁定目标

    x, y, _ = solution["center_arm"]
    print(
        f"目标：x={x:.4f} m，y={y:.4f} m，"
        f"B→C yaw={math.degrees(solution['tube_yaw']):.1f}°"
    )

    # 两个开关都没有时只做视觉预览：不会导入 CArm，更不会连接机械臂。
    if not (args.check_ik or args.execute):
        print("[PREVIEW ONLY] 没有连接机械臂。")
        return 0

    # --check-ik 和 --execute 都需要连接控制器；只有后者会调用 execute_plan。
    robot = SimpleCArm(args.ip)

    try:
        start = robot.current_joints()
        # 零位误差取六个关节绝对角度中的最大值，并从弧度转换成度。
        zero_error = math.degrees(float(np.max(np.abs(start))))
        print(f"起始零位最大关节误差：{zero_error:.3f}°")
        if zero_error > 3.0:
            raise RuntimeError(f"机械臂不在零位：最大关节误差 {zero_error:.2f}°")
        
        # 这里仅调用 IK/FK 搜索，不使能、不运动。
        plan = build_cap_up_plan(robot, solution, args.rotate_z)
        print(
            f"IK 通过：yaw 候选 {plan['yaw_index']}，"
            f"高度 {plan['rotate_z']:.3f} m，"
            f"盖朝上 twist {plan['twist_deg']:.0f}°"
        )
        print(
            "固定凹槽高空 TCP："
            f"({plan['groove_above_pose'][0]:.3f}, "
            f"{plan['groove_above_pose'][1]:.3f}, "
            f"{plan['groove_above_pose'][2]:.3f}) m"
        )
        print(
            "固定凹槽释放 TCP："
            f"({plan['groove_release_pose'][0]:.3f}, "
            f"{plan['groove_release_pose'][1]:.3f}, "
            f"{plan['groove_release_pose'][2]:.3f}) m"
        )

        if args.execute:
            execute_plan(robot, solution, plan)
        else:
            print("[IK CHECK ONLY] 没有使能、没有运动。")
    finally:
        # 规划失败、执行失败或正常完成，都确保断开机械臂网络连接。
        robot.close()
    return 0


if __name__ == "__main__":
    # 直接运行这个文件时 __name__ 才等于 "__main__"
    try:
        # SystemExit 把 main() 的返回值作为进程退出码交给操作系统。
        raise SystemExit(main())
    except KeyboardInterrupt:
        # ``from None`` 隐藏不必要的异常链，只显示给用户看的简短消息。
        raise SystemExit("\n用户取消。") from None
    
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"程序终止：{exc}") from None
