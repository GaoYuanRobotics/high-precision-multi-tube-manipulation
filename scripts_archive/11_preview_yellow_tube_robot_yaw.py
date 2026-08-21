#!/usr/bin/env python3
"""把黄色试管 B→C 图像方向转换为 CArm 基座 yaw 和目标四元数。

这个脚本用于验证“夹爪在黄色试管上方随试管方向旋转”的关键几何关系：

1. 在本文件内从 ``y-body`` / ``y-cap`` mask 计算中心、B、C。
2. 把中心、B、C 分别投影到机械臂基座平面。
3. 用基座坐标中的 B→C 计算试管 yaw。
4. 加上夹爪安装偏移，生成相差 180° 的两个等价候选姿态。
5. 把候选 yaw 转换为 CArm 使用的 ``[qx, qy, qz, qw]`` 四元数。

本脚本只预览和打印结果，不连接机械臂，也不发送运动或夹爪命令。
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml


# =============================================================================
# 1. 路径和稳定性参数
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = Path("/home/gaoyuan/camera_hand_calibration/config")

EXPECTED_CLASSES = {
    0: "p-body",
    1: "p-cap",
    2: "y-body",
    3: "y-cap",
}

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
CAMERA_FPS = 30
INFERENCE_SIZE = 1024
CONFIDENCE = 0.25
IOU = 0.70
DEVICE = "0"

TARGET_Z_M = 0.165
TARGET_QUATERNION = np.array(
    [0.999575504, 0.008135427, 0.027844000, 0.002709061],
    dtype=float,
)

STABLE_FRAMES = 10
MAX_CENTER_JITTER_PX = 3.0
MAX_ANGLE_JITTER_DEG = 3.0
MIN_TUBE_LENGTH_PX = 30.0
MIN_TUBE_ASPECT_RATIO = 2.0

GREEN = (80, 255, 80)
ORANGE = (0, 165, 255)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)


# =============================================================================
# 2. 模型、RealSense 和黄色试管二维几何
# =============================================================================

def find_latest_model() -> Path:
    """寻找 runs 中修改时间最新的 best.pt。"""

    candidates: list[Path] = []
    for path in (ROOT / "runs").rglob("best.pt"):
        if path.is_file():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "runs 下没有 best.pt，请使用 --model 指定模型"
        )
    latest_path = candidates[0]
    latest_time = latest_path.stat().st_mtime
    for path in candidates[1:]:
        modified_time = path.stat().st_mtime
        if modified_time > latest_time:
            latest_path = path
            latest_time = modified_time
    return latest_path.resolve()


def resolve_model(model_argument: str | None) -> str:
    """解析本地权重路径；省略时寻找最新 best.pt。"""

    if model_argument is None:
        return str(find_latest_model())
    candidate = Path(model_argument).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    return model_argument


def class_name(names, class_id: int) -> str:
    """兼容 Ultralytics 的 dict/list 两种类别表。"""

    if isinstance(names, dict):
        return str(names[class_id])
    return str(names[class_id])


def validate_model_classes(names) -> None:
    """要求模型类别 ID 和顺序严格等于项目的四个类别。"""

    actual: dict[int, str] = {}
    for index in range(len(names)):
        actual[index] = class_name(names, index)
    if actual != EXPECTED_CLASSES:
        raise ValueError(
            f"模型类别不匹配：期望 {EXPECTED_CLASSES}，实际 {actual}"
        )


def select_realsense_serial(rs, requested_serial: str | None) -> str:
    """只允许明确选择的设备，检测到多台相机时不随机猜测。"""

    devices = list(rs.context().query_devices())
    serials: list[str] = []
    for device in devices:
        serials.append(str(device.get_info(rs.camera_info.serial_number)))
    if requested_serial is not None:
        if requested_serial not in serials:
            raise RuntimeError(
                f"找不到 RealSense serial={requested_serial}；当前={serials}"
            )
        return requested_serial
    if len(serials) != 1:
        raise RuntimeError(
            f"检测到 {len(serials)} 台 RealSense，请用 --serial 指定：{serials}"
        )
    return serials[0]


class RealSenseColorSource:
    """打开一台确定的 RealSense，只读取 BGR 彩色流。"""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        serial: str | None,
    ) -> None:
        """初始化当前对象，并保存后续操作需要的状态。"""

        import pyrealsense2 as rs

        selected_serial = select_realsense_serial(rs, serial)
        self._pipeline = rs.pipeline()
        self._started = False
        config = rs.config()
        config.enable_device(selected_serial)
        config.enable_stream(
            rs.stream.color,
            width,
            height,
            rs.format.bgr8,
            fps,
        )
        try:
            profile = self._pipeline.start(config)
            self._started = True
            device = profile.get_device()
            actual_serial = str(
                device.get_info(rs.camera_info.serial_number)
            )
            if actual_serial != selected_serial:
                raise RuntimeError(
                    "实际启动相机与选择结果不一致："
                    f"{actual_serial} != {selected_serial}"
                )

            stream = profile.get_stream(
                rs.stream.color
            ).as_video_stream_profile()
            intrinsics = stream.get_intrinsics()
            self.intrinsics = {
                "fx": float(intrinsics.fx),
                "fy": float(intrinsics.fy),
                "ppx": float(intrinsics.ppx),
                "ppy": float(intrinsics.ppy),
            }
            name = device.get_info(rs.camera_info.name)
            self.description = (
                f"{name} serial={actual_serial} "
                f"color={stream.width()}x{stream.height()}@{stream.fps()}"
            )
        except Exception:
            if self._started:
                self._pipeline.stop()
                self._started = False
            raise

    def read(self):
        """等待并返回一张彩色图；偶发空帧返回 None。"""

        frames = self._pipeline.wait_for_frames(timeout_ms=5000)
        color = frames.get_color_frame()
        if color:
            return np.asanyarray(color.get_data())
        return None

    def close(self) -> None:
        """停止相机；允许重复调用。"""

        if self._started:
            self._pipeline.stop()
            self._started = False


def resize_mask(mask, width: int, height: int) -> np.ndarray:
    """把 YOLO mask 恢复到原始彩色图尺寸并转为布尔图。"""

    resized = cv2.resize(
        np.asarray(mask, dtype=np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized > 0.5


def largest_component(mask, name: str) -> np.ndarray:
    """只保留最大连通区域，拒绝空 mask。"""

    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _centers = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    if count <= 1:
        raise ValueError(f"{name} mask 为空")
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def mask_points(mask) -> np.ndarray:
    """把 mask 前景像素转换为 (x, y) 点集。"""

    rows, columns = np.nonzero(mask)
    return np.column_stack((columns, rows)).astype(np.float64)


def pca_axis(points: np.ndarray):
    """用 PCA 计算点集中心和单位长轴。"""

    center = points.mean(axis=0)
    _u, singular_values, vh = np.linalg.svd(
        points - center,
        full_matrices=False,
    )
    axis = vh[0]
    aspect = float(singular_values[0]) / max(
        float(singular_values[1]),
        1e-12,
    )
    return center, axis / np.linalg.norm(axis), aspect


def yellow_pose_from_masks(body_mask, cap_mask):
    """从黄色管身和管盖 mask 得到中心、管底 B、管盖侧端点 C。"""

    body = largest_component(body_mask, "y-body")
    cap = largest_component(cap_mask, "y-cap")
    points = mask_points(body)
    if len(points) < 10:
        raise ValueError("y-body 像素太少，无法稳定拟合长轴")

    # 第一次 PCA 后裁掉长轴两端各 5% 离群点，再重新拟合。
    first_center, first_axis, _aspect = pca_axis(points)
    projection = (points - first_center) @ first_axis
    lower, upper = np.percentile(projection, (5.0, 95.0))
    core = points[(projection >= lower) & (projection <= upper)]
    center, axis, aspect = pca_axis(core)
    if aspect < MIN_TUBE_ASPECT_RATIO:
        raise ValueError(
            f"y-body 不够细长：PCA 长短轴比 {aspect:.2f}"
        )

    projection = (core - center) @ axis
    end_a = center + float(projection.min()) * axis
    end_b = center + float(projection.max()) * axis
    tube_length = float(np.linalg.norm(end_b - end_a))
    if tube_length < MIN_TUBE_LENGTH_PX:
        raise ValueError(
            f"y-body 长轴过短：{tube_length:.1f} px"
        )

    cap_points = mask_points(cap)
    if len(cap_points) < 3:
        raise ValueError("y-cap 像素太少，无法判断 C 端")
    cap_center = cap_points.mean(axis=0)

    # 管盖必须靠近管身长轴及其中一个端点，避免错误 body/cap 配对。
    perpendicular = np.array([-axis[1], axis[0]])
    half_width = float(
        np.percentile(np.abs((core - center) @ perpendicular), 95)
    )
    perpendicular_error = abs(float((cap_center - center) @ perpendicular))
    max_perpendicular = max(
        3.0 * half_width,
        0.15 * tube_length,
        3.0,
    )
    distance_a = float(np.linalg.norm(cap_center - end_a))
    distance_b = float(np.linalg.norm(cap_center - end_b))
    if perpendicular_error > max_perpendicular:
        raise ValueError("y-cap 离 y-body 长轴太远")
    if min(distance_a, distance_b) > 0.35 * tube_length:
        raise ValueError("y-cap 离 y-body 两端都太远")
    if abs(distance_a - distance_b) < 0.20 * tube_length:
        raise ValueError("y-cap 不能明确区分管身的哪一端")

    if distance_a < distance_b:
        bottom, cap_end = end_b, end_a
    else:
        bottom, cap_end = end_a, end_b
    return center, bottom, cap_end


def find_yellow_bcg(result, image_shape):
    """从一帧 YOLO 结果提取唯一黄色管身和唯一黄色管盖。"""

    if result.boxes is None or result.masks is None:
        return None
    if len(result.boxes) == 0:
        return None

    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    names: list[str] = []
    for class_id in class_ids:
        names.append(class_name(result.names, class_id))
    body_indices: list[int] = []
    cap_indices: list[int] = []
    for index in range(len(names)):
        name = names[index]
        if name == "y-body":
            body_indices.append(index)
        if name == "y-cap":
            cap_indices.append(index)
    if len(body_indices) != 1 or len(cap_indices) != 1:
        return None

    masks = result.masks.data.detach().cpu().numpy()
    height, width = image_shape
    body_mask = resize_mask(masks[body_indices[0]], width, height)
    cap_mask = resize_mask(masks[cap_indices[0]], width, height)
    try:
        return yellow_pose_from_masks(body_mask, cap_mask)
    except ValueError:
        return None


# =============================================================================
# 3. 四元数：平面 yaw -> 夹爪向下姿态
# =============================================================================

def wrap_pi(angle_rad: float) -> float:
    """把角度限制到 [-pi, pi)。"""

    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def normalize_quaternion(quaternion) -> np.ndarray:
    """检查并归一化 CArm 的 [qx, qy, qz, qw] 四元数。"""

    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("四元数必须是四个有限数字 [qx, qy, qz, qw]")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("四元数长度不能为 0")
    return q / norm


def quaternion_multiply(left, right) -> np.ndarray:
    """计算 Hamilton 乘积，输入和输出顺序都是 [qx, qy, qz, qw]。"""

    ax, ay, az, aw = normalize_quaternion(left)
    bx, by, bz, bw = normalize_quaternion(right)
    return normalize_quaternion(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def yaw_quaternion(yaw_rad: float) -> np.ndarray:
    """生成绕机械臂基座 Z 轴旋转的四元数。"""

    half = float(yaw_rad) / 2.0
    return np.array([0.0, 0.0, math.sin(half), math.cos(half)])


def gripper_candidates(
    tube_yaw_rad: float,
    yaw_offset_rad: float,
    down_quaternion,
) -> list[dict]:
    """生成相差 180° 的两个平行夹爪候选姿态。

    ``down_quaternion`` 保留夹爪向下的 roll/pitch。左乘基座 Z 轴 yaw，
    只改变夹爪在工作台平面内的方向。
    """

    first_yaw = wrap_pi(tube_yaw_rad + yaw_offset_rad)
    candidate_yaws = [first_yaw, wrap_pi(first_yaw + math.pi)]

    candidates = []
    for yaw_index in range(len(candidate_yaws)):
        index = yaw_index + 1
        yaw_rad = candidate_yaws[yaw_index]
        quaternion = quaternion_multiply(
            yaw_quaternion(yaw_rad),
            down_quaternion,
        )
        candidates.append(
            {
                "index": index,
                "yaw_rad": yaw_rad,
                "quaternion": quaternion,
            }
        )
    return candidates


# =============================================================================
# 4. 手眼矩阵和像素投影
# =============================================================================

def load_matrix(filename: str, shape: tuple[int, int]) -> np.ndarray:
    """读取最终标定目录中的矩阵。"""

    matrix = np.loadtxt(
        CALIBRATION_DIR / filename,
        delimiter=",",
        dtype=float,
    )
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"{filename} 不是有效的 {shape} 矩阵")
    return matrix


def load_calibration() -> dict:
    """一次性读取手眼矩阵，并检查三份矩阵一致。"""

    required_files = (
        "camera.yaml",
        "intrinsic.txt",
        "T_cam2ws.txt",
        "T_arm2ws.txt",
        "T_arm2cam.txt",
    )
    missing: list[str] = []
    for name in required_files:
        if not (CALIBRATION_DIR / name).is_file():
            missing.append(name)
    if missing:
        raise FileNotFoundError(
            f"手眼标定目录缺少：{', '.join(missing)}；"
            f"目录={CALIBRATION_DIR}"
        )

    with (CALIBRATION_DIR / "camera.yaml").open(
        "r",
        encoding="utf-8",
    ) as file:
        camera = yaml.safe_load(file)

    intrinsic = load_matrix("intrinsic.txt", (3, 3))
    camera_from_workspace = load_matrix("T_cam2ws.txt", (4, 4))
    arm_from_workspace = load_matrix("T_arm2ws.txt", (4, 4))
    arm_from_camera = load_matrix("T_arm2cam.txt", (4, 4))

    expected = arm_from_workspace @ np.linalg.inv(camera_from_workspace)
    matrix_error = float(np.max(np.abs(expected - arm_from_camera)))
    if matrix_error > 1e-3:
        raise ValueError(
            "T_cam2ws、T_arm2ws 和 T_arm2cam 不一致："
            f"最大组合误差 {matrix_error:.6f}"
        )

    return {
        "intrinsic": intrinsic,
        "camera_from_workspace": camera_from_workspace,
        "arm_from_workspace": arm_from_workspace,
        "width": int(camera["img_width"]),
        "height": int(camera["img_height"]),
        "matrix_error": matrix_error,
    }


def check_camera_intrinsics(source, calibration: dict) -> None:
    """确认实时彩色流与手眼标定使用同一套内参。"""

    intrinsic = calibration["intrinsic"]
    saved = np.array(
        [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]]
    )
    live = np.array(
        [
            source.intrinsics["fx"],
            source.intrinsics["fy"],
            source.intrinsics["ppx"],
            source.intrinsics["ppy"],
        ]
    )
    max_error_px = float(np.max(np.abs(saved - live)))
    if max_error_px > 1.0:
        raise ValueError(
            "实时 RealSense 内参与手眼标定内参不一致："
            f"最大差值 {max_error_px:.3f} px"
        )
    print(f"相机内参检查通过：最大差值 {max_error_px:.3f} px")


def pixel_to_arm_plane(point_xy, calibration: dict) -> np.ndarray:
    """把一个彩色图像像素投影到机械臂基座中的标定平面，单位为米。"""

    u, v = np.asarray(point_xy, dtype=float)
    width = calibration["width"]
    height = calibration["height"]
    if not np.isfinite([u, v]).all() or not (0 <= u < width and 0 <= v < height):
        raise ValueError(
            f"像素 ({u:.3f}, {v:.3f}) 超出 0≤u<{width}, 0≤v<{height}"
        )

    intrinsic = calibration["intrinsic"]
    camera_from_workspace = calibration["camera_from_workspace"]
    arm_from_workspace = calibration["arm_from_workspace"]

    # 像素先变成相机射线，再和工作台 Z=0 平面求交。
    ray_camera = np.linalg.solve(intrinsic, np.array([u, v, 1.0]))
    workspace_from_camera = np.linalg.inv(camera_from_workspace)
    camera_origin_workspace = workspace_from_camera[:3, 3]
    ray_workspace = workspace_from_camera[:3, :3] @ ray_camera
    if abs(ray_workspace[2]) < 1e-9:
        raise ValueError("像素射线与工作台平面平行")

    scale = -camera_origin_workspace[2] / ray_workspace[2]
    if scale <= 0:
        raise ValueError("像素对应点位于相机后方")

    point_workspace_mm = camera_origin_workspace + scale * ray_workspace
    point_workspace_mm[2] = 0.0
    point_arm_mm = (
        arm_from_workspace @ np.append(point_workspace_mm, 1.0)
    )[:3]
    return point_arm_mm / 1000.0


def calculate_robot_yaw_solution(
    center_xy,
    bottom_xy,
    cap_xy,
    calibration: dict,
    yaw_offset_deg: float,
    down_quaternion,
    target_z_m: float,
) -> dict:
    """把图像中心/B/C 转换为基座 yaw、候选四元数和候选 TCP Pose。"""

    center_arm = pixel_to_arm_plane(center_xy, calibration)
    bottom_arm = pixel_to_arm_plane(bottom_xy, calibration)
    cap_arm = pixel_to_arm_plane(cap_xy, calibration)

    base_axis_xy = cap_arm[:2] - bottom_arm[:2]
    base_axis_length_m = float(np.linalg.norm(base_axis_xy))
    if base_axis_length_m < 1e-6:
        raise ValueError("转换后的 B/C 距离过小，无法计算机械臂基座 yaw")

    image_axis = np.asarray(cap_xy, dtype=float) - np.asarray(
        bottom_xy,
        dtype=float,
    )
    image_angle_rad = math.atan2(image_axis[1], image_axis[0])
    tube_yaw_rad = math.atan2(base_axis_xy[1], base_axis_xy[0])
    candidates = gripper_candidates(
        tube_yaw_rad,
        math.radians(yaw_offset_deg),
        down_quaternion,
    )

    for candidate in candidates:
        quaternion = candidate["quaternion"]
        candidate["target_pose"] = np.array(
            [
                center_arm[0],
                center_arm[1],
                float(target_z_m),
                quaternion[0],
                quaternion[1],
                quaternion[2],
                quaternion[3],
            ],
            dtype=float,
        )

    return {
        "center_xy": np.asarray(center_xy, dtype=float),
        "bottom_xy": np.asarray(bottom_xy, dtype=float),
        "cap_xy": np.asarray(cap_xy, dtype=float),
        "center_arm_m": center_arm,
        "bottom_arm_m": bottom_arm,
        "cap_arm_m": cap_arm,
        "image_angle_rad": image_angle_rad,
        "tube_yaw_rad": tube_yaw_rad,
        "base_axis_length_m": base_axis_length_m,
        "yaw_offset_deg": float(yaw_offset_deg),
        "candidates": candidates,
    }


# =============================================================================
# 5. 连续帧稳定检查
# =============================================================================

def angle_difference_rad(first: float, second: float) -> float:
    """返回两个有方向角之间的最小差值。"""

    return abs(wrap_pi(first - second))


def stable_bcg(recent_geometries):
    """中心和 B→C 角度都连续稳定时，返回各点的中位数。"""

    if len(recent_geometries) < STABLE_FRAMES:
        return None

    center_values: list[np.ndarray] = []
    bottom_values: list[np.ndarray] = []
    cap_values: list[np.ndarray] = []
    for item in recent_geometries:
        center_values.append(item[0])
        bottom_values.append(item[1])
        cap_values.append(item[2])
    centers = np.asarray(center_values)
    bottoms = np.asarray(bottom_values)
    caps = np.asarray(cap_values)

    center_median = np.median(centers, axis=0)
    center_jitter = np.linalg.norm(centers - center_median, axis=1)
    if float(np.max(center_jitter)) > MAX_CENTER_JITTER_PX:
        return None

    axes = caps - bottoms
    angles = np.arctan2(axes[:, 1], axes[:, 0])
    mean_angle = math.atan2(
        float(np.mean(np.sin(angles))),
        float(np.mean(np.cos(angles))),
    )
    angle_errors: list[float] = []
    for angle in angles:
        angle_errors.append(angle_difference_rad(float(angle), mean_angle))
    max_angle_error = max(angle_errors)
    if math.degrees(max_angle_error) > MAX_ANGLE_JITTER_DEG:
        return None

    return (
        center_median,
        np.median(bottoms, axis=0),
        np.median(caps, axis=0),
    )


# =============================================================================
# 6. 画面显示和终端报告
# =============================================================================

def draw_preview(image, geometry, stable_geometry, solution) -> np.ndarray:
    """在 YOLO 结果上画 B、C、中心和转换后的角度。"""

    output = image.copy()
    if geometry is not None:
        center, bottom, cap = geometry
        center_px = tuple(np.round(center).astype(int))
        bottom_px = tuple(np.round(bottom).astype(int))
        cap_px = tuple(np.round(cap).astype(int))
        cv2.line(output, bottom_px, cap_px, YELLOW, 3, cv2.LINE_AA)
        cv2.circle(output, center_px, 7, GREEN, -1)
        cv2.circle(output, bottom_px, 7, BLUE, -1)
        cv2.circle(output, cap_px, 7, RED, -1)
        cv2.putText(
            output,
            "B",
            (bottom_px[0] + 8, bottom_px[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            BLUE,
            2,
        )
        cv2.putText(
            output,
            "C",
            (cap_px[0] + 8, cap_px[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            RED,
            2,
        )

    if stable_geometry is None:
        lines = [
            "Waiting for stable yellow B/C...",
            "Need one y-body + one matched y-cap",
            "Q/ESC: quit",
        ]
        color = ORANGE
    else:
        candidate_1, candidate_2 = solution["candidates"]
        lines = [
            "STABLE - press C to print result",
            (
                "IMAGE B->C="
                f"{math.degrees(solution['image_angle_rad']):.1f} deg"
            ),
            (
                "BASE TUBE YAW="
                f"{math.degrees(solution['tube_yaw_rad']):.1f} deg"
            ),
            (
                f"GRIPPER YAW A/B="
                f"{math.degrees(candidate_1['yaw_rad']):.1f}/"
                f"{math.degrees(candidate_2['yaw_rad']):.1f} deg"
            ),
            "C: lock/print    Q/ESC: quit",
        ]
        color = GREEN

    for index in range(len(lines)):
        line = lines[index]
        cv2.putText(
            output,
            line,
            (20, 35 + index * 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def print_solution(solution) -> None:
    """把离线或实时锁定的转换结果完整打印到终端。"""

    print("\n========== 黄色试管动态 yaw 预览 ==========")
    print(
        "中心像素："
        f"{np.round(solution['center_xy'], 3).tolist()}"
    )
    print(f"B 像素：{np.round(solution['bottom_xy'], 3).tolist()}")
    print(f"C 像素：{np.round(solution['cap_xy'], 3).tolist()}")
    print(
        "图像 B→C 角度："
        f"{math.degrees(solution['image_angle_rad']):.3f}°"
    )
    print(
        "机械臂基座 B→C yaw："
        f"{math.degrees(solution['tube_yaw_rad']):.3f}°"
    )
    print(f"夹爪 yaw 偏移：{solution['yaw_offset_deg']:.3f}°")
    print(
        "中心对应机械臂平面点 m："
        f"{np.round(solution['center_arm_m'], 6).tolist()}"
    )

    for candidate in solution["candidates"]:
        print(f"\n候选 {candidate['index']}：")
        print(f"  夹爪 yaw：{math.degrees(candidate['yaw_rad']):.3f}°")
        print(
            "  四元数 [qx,qy,qz,qw]："
            f"{np.round(candidate['quaternion'], 9).tolist()}"
        )
        print(
            "  目标 Pose [x,y,z,qx,qy,qz,qw]："
            f"{np.round(candidate['target_pose'], 9).tolist()}"
        )

    print("\n说明：两个候选相差 180°，本脚本不会替你选择 IK。")
    print("说明：yaw-offset 尚需通过安全高空对齐实验确认。")
    print("[PREVIEW ONLY] 没有连接机械臂，也没有发送任何命令。")
    print("============================================\n")


# =============================================================================
# 7. RealSense 实时预览
# =============================================================================

def run_realtime(args, calibration):
    """持续检测，直到 B/C 和中心稳定后按 C 锁定。"""

    model_path = resolve_model(args.model)

    # 延迟导入，确保 --help 和 --pixels 不需要加载 YOLO。
    from ultralytics import YOLO

    print(f"模型：{model_path}")
    model = YOLO(model_path)
    validate_model_classes(model.names)

    source = RealSenseColorSource(
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        CAMERA_FPS,
        args.serial,
    )
    window_name = "11 - yellow tube robot yaw"
    recent_geometries = deque(maxlen=STABLE_FRAMES)

    try:
        check_camera_intrinsics(source, calibration)
        print(f"相机：{source.description}")
        print("中心与 B→C 角度同时稳定后按 C；按 Q 或 Esc 退出。")
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            frame = source.read()
            if frame is None:
                continue

            result = model.predict(
                source=frame,
                imgsz=INFERENCE_SIZE,
                conf=CONFIDENCE,
                iou=IOU,
                device=DEVICE,
                # 项目视觉脚本统一固定使用 FP32。
                half=False,
                max_det=50,
                retina_masks=True,
                verbose=False,
            )[0]

            height, width = frame.shape[:2]
            geometry = find_yellow_bcg(
                result,
                (height, width),
            )
            if geometry is None:
                recent_geometries.clear()
            else:
                recent_geometries.append(geometry)

            stable_geometry = stable_bcg(recent_geometries)
            solution = None
            if stable_geometry is not None:
                center_xy, bottom_xy, cap_xy = stable_geometry
                solution = calculate_robot_yaw_solution(
                    center_xy,
                    bottom_xy,
                    cap_xy,
                    calibration,
                    args.yaw_offset_deg,
                    TARGET_QUATERNION,
                    TARGET_Z_M,
                )

            shown = result.plot(
                conf=True,
                labels=True,
                boxes=True,
                masks=True,
                color_mode="class",
            )
            shown = draw_preview(
                shown,
                geometry,
                stable_geometry,
                solution,
            )
            cv2.imshow(window_name, shown)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                raise KeyboardInterrupt
            if key in (ord("c"), ord("C")) and solution is not None:
                return solution
    finally:
        source.close()
        cv2.destroyAllWindows()


# =============================================================================
# 8. 命令行入口
# =============================================================================

def parse_args():
    """读取终端参数，并返回 argparse.Namespace。"""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        default=None,
        help="YOLO Seg 权重路径；省略时寻找 runs 下最新 best.pt。",
    )
    parser.add_argument(
        "--serial",
        default=None,
        help="RealSense 序列号；连接多台相机时必须指定。",
    )
    parser.add_argument(
        "--yaw-offset-deg",
        type=float,
        default=0.0,
        help="试管方向到夹爪参考方向的偏移角；默认 0°，必须现场标定。",
    )
    parser.add_argument(
        "--pixels",
        nargs=6,
        type=float,
        metavar=("CENTER_U", "CENTER_V", "B_U", "B_V", "C_U", "C_V"),
        help="跳过 RealSense，直接用中心/B/C 像素做离线计算。",
    )
    return parser.parse_args()


def main() -> int:
    """按照本脚本的编号流程依次执行各个步骤。"""

    args = parse_args()
    if not math.isfinite(args.yaw_offset_deg):
        raise ValueError("--yaw-offset-deg 必须是有限数字")

    calibration = load_calibration()
    print(
        "手眼矩阵组合检查通过："
        f"最大差值 {calibration['matrix_error']:.6f}"
    )

    if args.pixels is None:
        solution = run_realtime(args, calibration)
    else:
        values = np.asarray(args.pixels, dtype=float)
        center = values[0:2]
        bottom = values[2:4]
        cap = values[4:6]
        solution = calculate_robot_yaw_solution(
            center,
            bottom,
            cap,
            calibration,
            args.yaw_offset_deg,
            TARGET_QUATERNION,
            TARGET_Z_M,
        )

    print_solution(solution)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("\n用户取消，窗口已经关闭。") from None
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"程序终止：{exc}") from None
