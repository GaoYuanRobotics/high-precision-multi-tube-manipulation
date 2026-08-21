#!/usr/bin/env python3
"""检测黄色试管中心和角度，并可让 CArm 到试管上方做闭爪测试。

本脚本把视觉定位、手眼坐标转换和第一版 CArm 动作连在一起：

1. 打开 RealSense 彩色流。
2. 使用训练好的 YOLO Seg 模型检测 ``y-body``。
3. 根据黄色管身 mask 计算中心点 ``center_xy``。
4. 检测到可靠 ``y-cap`` 时，显示从管底 B 指向管盖端 C 的图像角度。
5. 中心点连续稳定 10 帧后显示绿色 ``STABLE``。
6. 按 C 键锁定并打印中心与角度，按 Q 或 Esc 退出。

默认只做视觉、手眼坐标转换和动作计划预览。只有添加 ``--execute`` 并输入
与本次目标一致的确认文字后，才会连接 CArm。

机械臂第一版动作：

    检查零位 -> 准备位 -> 开爪 -> 黄色试管中心上方
    -> 等待 3 秒 -> 空中闭爪 -> 闭爪后抬高位
    -> 试管竖直位 -> 竖直后目标位 -> 松爪下降位
    -> 松开夹爪 -> 松爪后目标位 -> 关节零位

当前不会下降到试管抓取高度；最后只会下降到固定松爪位并打开夹爪。
松爪后会移动到固定目标位，最后回到关节零位。

初学者阅读建议：先看文件末尾的 ``main()`` 了解入口，再看
``execute_demo()`` 的 1/11～11/11 主流程。固定点位运动都调用
``move_to_fixed_target()``，它统一完成路径检查、运动和到位验收。
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml


# =============================================================================
# 1. 初学者最常调整的参数
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

# 最终手眼标定文件所在目录。
CALIBRATION_DIR = Path("/home/gaoyuan/camera_hand_calibration/config")

# 用户已经确认：安装夹爪后使用 tool=1。
TOOL_INDEX = 1

# 黄色试管中心上方的固定 TCP 高度和姿态。
TARGET_Z_M = 0.165
TARGET_QUATERNION = np.array(
    [0.999575504, 0.008135427, 0.027844000, 0.002709061],
    dtype=float,
)

# 所有普通运动和到位结果共用的安全限制。
MAX_TRAVEL_M = 0.300
MAX_TILT_DEG = 10.0
MAX_JOINT_ERROR_DEG = 3.0
MAX_POSITION_ERROR_MM = 2.0
MAX_ORIENTATION_ERROR_DEG = 1.0
MIN_READY_Z_M = TARGET_Z_M + 0.030

# 安装夹爪后的垂直向下准备位，以及六关节零位。
DOWN_READY_JOINTS = np.array(
    [-0.001726, 1.751210, -0.626573, -0.000954, 0.446518, -0.000954],
    dtype=float,
)
ZERO_JOINTS = np.zeros(6, dtype=float)

# RealSense 彩色流参数。
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
CAMERA_FPS = 30
MODEL_CLASSES = ("p-body", "p-cap", "y-body", "y-cap")

# YOLO 推理参数。
INFERENCE_SIZE = 1024
CONFIDENCE = 0.25
IOU = 0.70
DEVICE = "0"

# 最近 10 帧中心点到中位数的距离都不超过 3 像素，才认为稳定。
STABLE_FRAMES = 10
MAX_JITTER_PX = 3.0

# OpenCV 使用 BGR；绿色用于标出黄色管身中心。
CENTER_COLOR_BGR = (0, 255, 0)

# 夹爪和第一版悬停参数。
WAIT_SECONDS = 3.0
GRIPPER_OPEN_M = 0.060
GRIPPER_CLOSE_M = 0.000
GRIPPER_FORCE_N = 10.0

# 目标上方到固定抬高位是一段已由现场验证的特定运动。
# TARGET_Z_M 降到 0.165 m 后，该段 TCP 直线距离约为 0.302 m，
# 因此只给这一段使用 0.310 m 上限；其他运动仍使用通用 0.300 m 上限。
MAX_TARGET_TO_LIFT_M = 0.310

# 闭爪后移动到的抬高姿态，来自用户在 CArm 网页控制端确认的新截图。
# 这里保存的是六个关节角，单位都是 rad。
POST_CLOSE_LIFT_JOINTS = np.array(
    [-0.024351, 1.437900, -1.388000, -0.004005, 1.463910, -0.041772],
    dtype=float,
)

# 截图中同时显示的 TCP 位置，仅用于日志说明，不作为运动命令。
POST_CLOSE_REFERENCE_TCP_M = np.array(
    [0.205338, -0.005431, 0.462922],
    dtype=float,
)

# 从闭爪后抬高位直接移动到的“试管竖直位”。
# 用户已确认：当前夹持关系中，夹爪水平基本等于试管竖直。
TUBE_VERTICAL_JOINTS = np.array(
    [-0.019750, 1.185170, -0.345045, -0.018883, -0.829137, -0.032998],
    dtype=float,
)

# 网页截图显示的竖直位 TCP，只用于运行时输出和人工核对。
TUBE_VERTICAL_REFERENCE_TCP_M = np.array(
    [0.196661, -0.002381, 0.352599],
    dtype=float,
)

# 这一段已由用户确认可以直接过去。J5 的预期变化约为 131.4°，
# 因此只给“抬高位 -> 试管竖直位”设置单独的 135° 上限。
MAX_DIRECT_VERTICAL_JOINT_CHANGE_DEG = 135.0
MIN_VERTICAL_TCP_Z_M = 0.300
MIN_VERTICAL_TOOL_TILT_DEG = 75.0
MAX_VERTICAL_TOOL_TILT_DEG = 105.0

# 到达试管竖直位后，先移动到用户上一张截图确认的目标姿态。
POST_VERTICAL_TARGET_JOINTS = np.array(
    [-0.743011, 1.892710, -0.322156, -0.275617, -1.478030, -0.050546],
    dtype=float,
)

# 上一张截图显示的参考 TCP，只用于日志和人工核对。
POST_VERTICAL_REFERENCE_TCP_M = np.array(
    [0.235773, -0.176849, 0.237061],
    dtype=float,
)

# 竖直后目标位的最低允许高度。
MIN_POST_VERTICAL_TCP_Z_M = 0.200

# 到达上面的竖直后目标位以后，再下降到最新截图中的松爪位置。
RELEASE_LOWER_JOINTS = np.array(
    [-0.744928, 1.985130, -0.288205, -0.275998, -1.570800, -0.085642],
    dtype=float,
)

# 最新下降位置的截图参考 TCP，只用于日志和人工核对。
RELEASE_LOWER_REFERENCE_TCP_M = np.array(
    [0.234552, -0.176262, 0.214298],
    dtype=float,
)

# 松爪位仍必须高于 0.200 m。
MIN_RELEASE_TCP_Z_M = 0.200

# 松开夹爪以后，再移动到用户最新截图中的固定目标位。
POST_RELEASE_TARGET_JOINTS = np.array(
    [-0.745312, 1.620440, -0.025749, -0.275998, -1.567290, -0.084498],
    dtype=float,
)

# 最新截图显示的 TCP，只用于日志和人工核对。
POST_RELEASE_REFERENCE_TCP_M = np.array(
    [0.162846, -0.110238, 0.259891],
    dtype=float,
)

# 松爪后目标位必须保持在安全高度以上。
MIN_POST_RELEASE_TCP_Z_M = 0.200

# 松爪后目标位直接回零的最大关节变化约为 92.84°。
# 只为最后这一个固定过渡使用 95° 上限。
MAX_RETURN_ZERO_JOINT_CHANGE_DEG = 95.0
MIN_ZERO_TCP_Z_M = 0.200


# =============================================================================
# 2. 本脚本自己的 RealSense、模型类别和黄色试管几何
# =============================================================================

def latest_model() -> str:
    """返回 runs 目录中修改时间最新的 best.pt。"""

    weights: list[Path] = []
    for path in (ROOT / "runs").rglob("best.pt"):
        if path.is_file():
            weights.append(path)
    if not weights:
        raise FileNotFoundError("runs 下没有 best.pt，请用 --model 指定权重")
    latest_weight = weights[0]
    latest_time = latest_weight.stat().st_mtime
    for path in weights[1:]:
        modified_time = path.stat().st_mtime
        if modified_time > latest_time:
            latest_weight = path
            latest_time = modified_time
    return str(latest_weight)


def ordered_model_names(names) -> tuple[str, ...]:
    """把 Ultralytics 类别名称整理成按类别 ID 排序的元组。"""

    if isinstance(names, dict):
        ordered_names: list[str] = []
        for index in sorted(names):
            ordered_names.append(str(names[index]))
        return tuple(ordered_names)
    ordered_names = []
    for name in names:
        ordered_names.append(str(name))
    return tuple(ordered_names)


class RealSenseColorSource:
    """只打开 RealSense 彩色流，并保存实时内参。"""

    def __init__(self, width: int, height: int, fps: int, serial: str | None):
        """初始化当前对象，并保存后续操作需要的状态。"""

        import pyrealsense2 as rs

        devices = list(rs.context().query_devices())
        serials: list[str] = []
        for device in devices:
            serials.append(str(device.get_info(rs.camera_info.serial_number)))
        if serial:
            if serial not in serials:
                raise ValueError(f"指定的 RealSense 不存在：{serial}；当前设备：{serials}")
            selected = serial
        elif len(serials) == 1:
            selected = serials[0]
        else:
            raise ValueError(f"检测到 {len(serials)} 台 RealSense，请用 --serial 指定：{serials}")

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(selected)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self.pipeline.start(config)
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        value = stream.get_intrinsics()
        self.intrinsics = {
            "fx": value.fx,
            "fy": value.fy,
            "ppx": value.ppx,
            "ppy": value.ppy,
        }
        self.description = (
            f"Intel RealSense serial={selected} "
            f"color={stream.width()}x{stream.height()}@{stream.fps()}"
        )

    def read(self):
        """读取一帧数据；暂时没有有效帧时返回 None。"""

        frame = self.pipeline.wait_for_frames(5000).get_color_frame()
        if frame:
            return np.asanyarray(frame.get_data())
        return None

    def close(self) -> None:
        """释放当前对象占用的相机、文件或连接资源。"""

        self.pipeline.stop()


def largest_component(mask: np.ndarray) -> np.ndarray:
    """只保留二值掩膜中面积最大的连通区域。"""

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count < 2:
        raise ValueError("黄色试管掩膜为空")
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == index


def check_camera_intrinsics(source) -> None:
    """确认实时彩色流内参与手眼标定使用的内参一致。"""

    intrinsic = np.loadtxt(
        CALIBRATION_DIR / "intrinsic.txt",
        delimiter=",",
        dtype=float,
    )
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("intrinsic.txt 不是有效的 3×3 内参矩阵")

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


def median_if_stable(points) -> tuple[float, float] | None:
    """中心点连续稳定时返回中位数，否则返回 None。"""

    if len(points) < STABLE_FRAMES:
        return None

    array = np.asarray(points, dtype=float)
    median = np.median(array, axis=0)
    distances = np.linalg.norm(array - median, axis=1)
    if float(np.max(distances)) > MAX_JITTER_PX:
        return None

    return float(median[0]), float(median[1])


def find_yellow_geometry(
    result,
    image_shape,
):
    """计算黄色管身中心和由黄色管盖确定的 B→C 图像角度。

    ``center`` 只来自 y-body。``angle_deg`` 还要求 y-cap 唯一且能与管身
    端点可靠配对；否则中心仍可显示，但角度返回 None。
    """

    if result.boxes is None or result.masks is None:
        return None, None

    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    bodies = np.flatnonzero(class_ids == 2)
    caps = np.flatnonzero(class_ids == 3)
    if len(bodies) != 1:
        return None, None

    masks = result.masks.data.detach().cpu().numpy()
    height, width = image_shape

    def points(index: int) -> np.ndarray:
        """把一个实例掩膜转换为原图坐标中的前景点。"""

        mask = cv2.resize(masks[index], (width, height)) > 0.5
        ys, xs = np.nonzero(largest_component(mask))
        return np.column_stack((xs, ys)).astype(float)

    body = points(int(bodies[0]))
    if len(body) < 30:
        return None, None
    center = np.mean(body, axis=0)

    # 没有唯一管盖时仍返回中心，只是不输出有方向角度。
    if len(caps) != 1:
        return center, None
    cap = points(int(caps[0]))
    if len(cap) < 5:
        return center, None

    values, vectors = np.linalg.eigh(np.cov((body - center).T))
    axis = vectors[:, int(np.argmax(values))]
    projections = (body - center) @ axis
    end_1 = center + np.percentile(projections, 1) * axis
    end_2 = center + np.percentile(projections, 99) * axis
    if np.linalg.norm(end_2 - end_1) < 30:
        return center, None
    cap_center = np.mean(cap, axis=0)
    if np.linalg.norm(cap_center - end_1) < np.linalg.norm(cap_center - end_2):
        bottom, cap_end = end_2, end_1
    else:
        bottom, cap_end = end_1, end_2
    angle_deg = math.degrees(
        math.atan2(float(cap_end[1] - bottom[1]), float(cap_end[0] - bottom[0]))
    )
    return center, angle_deg


def draw_center(image, center, stable_center, angle_deg) -> np.ndarray:
    """绘制黄色管身中心、B→C 角度、稳定状态和操作提示。"""

    output = image.copy()

    if center is not None:
        u, v = np.round(center).astype(int)
        # 圆点标出中心，十字线便于观察单像素位置。
        cv2.circle(output, (u, v), 7, CENTER_COLOR_BGR, -1)
        cv2.line(output, (u - 14, v), (u + 14, v), CENTER_COLOR_BGR, 2)
        cv2.line(output, (u, v - 14), (u, v + 14), CENTER_COLOR_BGR, 2)
        cv2.putText(
            output,
            f"Y-CENTER ({center[0]:.1f}, {center[1]:.1f})",
            (u + 12, v - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            CENTER_COLOR_BGR,
            2,
            cv2.LINE_AA,
        )

    if stable_center is None:
        status = "Detecting one yellow tube body..."
        status_color = (0, 165, 255)
    else:
        status = (
            f"STABLE CENTER=({stable_center[0]:.1f}, "
            f"{stable_center[1]:.1f}) - press C"
        )
        status_color = CENTER_COLOR_BGR

    angle_text = "Y-ANGLE: waiting for one matched y-cap"
    angle_color = (0, 165, 255)
    if angle_deg is not None:
        angle_text = f"Y-ANGLE B->C: {angle_deg:.1f} deg"
        angle_color = CENTER_COLOR_BGR
    cv2.putText(
        output,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        angle_text,
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        angle_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "C: lock center    Q/ESC: quit",
        (20, 101),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


# =============================================================================
# 3. RealSense 实时循环
# =============================================================================

def run_preview(args) -> tuple[tuple[float, float], float | None]:
    """持续检测黄色管身，直到用户按 C 锁定稳定中心点。"""

    model_path = args.model or latest_model()

    # 延迟导入：运行 --help 时不会加载 YOLO。
    from ultralytics import YOLO

    print(f"模型：{model_path}")
    model = YOLO(model_path)
    if str(model.task) != "segment" or ordered_model_names(model.names) != MODEL_CLASSES:
        raise ValueError("模型必须是 p-body、p-cap、y-body、y-cap 四类分割模型")

    source = RealSenseColorSource(
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        CAMERA_FPS,
        args.serial,
    )
    window_name = "10 - yellow tube center"
    recent_centers = deque(maxlen=STABLE_FRAMES)

    try:
        check_camera_intrinsics(source)
        print(f"相机：{source.description}")
        print("目标类别：y-body")
        print("中心稳定后按 C 锁定；按 Q 或 Esc 退出。")
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
            center, angle_deg = find_yellow_geometry(
                result,
                (height, width),
            )

            if center is None:
                # 当前帧无唯一、有效黄色管身时，重新累计连续稳定帧。
                recent_centers.clear()
            else:
                recent_centers.append(center)

            stable_center = median_if_stable(recent_centers)

            # result.plot 负责绘制 YOLO mask、框和类别；本脚本只额外画中心点。
            shown = result.plot(
                conf=True,
                labels=True,
                boxes=True,
                masks=True,
                color_mode="class",
            )
            shown = draw_center(shown, center, stable_center, angle_deg)

            cv2.imshow(window_name, shown)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                raise KeyboardInterrupt
            if key in (ord("c"), ord("C")) and stable_center is not None:
                return stable_center, angle_deg
    finally:
        source.close()
        cv2.destroyAllWindows()


# =============================================================================
# 4. 黄色中心像素转换为 CArm 上方目标
# =============================================================================

def load_matrix(filename, shape):
    """从最终标定目录读取矩阵，并检查形状和有限值。"""

    matrix = np.loadtxt(
        CALIBRATION_DIR / filename,
        delimiter=",",
        dtype=float,
    )
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"{filename} 不是有效的 {shape} 矩阵")
    return matrix


def pixel_to_target_pose(u, v):
    """把工作台平面上的像素转换为 CArm 基座坐标系目标。"""

    with (CALIBRATION_DIR / "camera.yaml").open(
        "r",
        encoding="utf-8",
    ) as file:
        camera = yaml.safe_load(file)

    width = int(camera["img_width"])
    height = int(camera["img_height"])
    if not (0 <= u < width and 0 <= v < height):
        raise ValueError(f"像素超出图像范围：0≤u<{width}, 0≤v<{height}")

    intrinsic = load_matrix("intrinsic.txt", (3, 3))
    camera_from_workspace = load_matrix("T_cam2ws.txt", (4, 4))
    arm_from_workspace = load_matrix("T_arm2ws.txt", (4, 4))

    # 像素先变成相机射线，再与工作台 Z=0 平面求交。
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

    lift_mm = TARGET_Z_M * 1000.0 - point_arm_mm[2]
    if lift_mm <= 0:
        raise ValueError(
            f"固定 Z={TARGET_Z_M * 1000:.0f} mm 不高于标定接触位置"
        )

    quaternion_values = TARGET_QUATERNION.tolist()
    target_pose = [
        float(point_arm_mm[0] / 1000.0),
        float(point_arm_mm[1] / 1000.0),
        TARGET_Z_M,
        quaternion_values[0],
        quaternion_values[1],
        quaternion_values[2],
        quaternion_values[3],
    ]
    return target_pose, point_workspace_mm, point_arm_mm, lift_mm


def pixel_to_target(center):
    """检查手眼矩阵，并把黄色中心像素转换为 CArm 上方目标 Pose。"""

    camera_from_workspace = load_matrix("T_cam2ws.txt", (4, 4))
    arm_from_workspace = load_matrix("T_arm2ws.txt", (4, 4))
    arm_from_camera = load_matrix("T_arm2cam.txt", (4, 4))

    # 三份矩阵必须满足 arm<-camera = arm<-workspace @ workspace<-camera。
    expected = arm_from_workspace @ np.linalg.inv(camera_from_workspace)
    matrix_error = float(np.max(np.abs(expected - arm_from_camera)))
    if matrix_error > 1e-3:
        raise ValueError(
            "T_cam2ws、T_arm2ws 和 T_arm2cam 不一致："
            f"最大组合误差 {matrix_error:.6f}"
        )

    return pixel_to_target_pose(center[0], center[1])


def print_plan(
    center,
    angle_deg,
    target_pose,
    workspace_point,
    arm_point,
) -> None:
    """在连接机械臂之前，完整打印视觉结果和动作计划。"""

    print("\n========== 黄色试管上方闭爪计划 ==========")
    print(f"中心像素：(u={center[0]:.2f}, v={center[1]:.2f})")
    if angle_deg is None:
        print("图像角度：未确定（没有唯一、可靠配对的 y-cap）")
    else:
        print(f"图像 B→C 角度：{angle_deg:.2f}°")
    print(f"工作台交点 mm：{np.round(workspace_point, 3).tolist()}")
    print(f"机械臂标定平面点 mm：{np.round(arm_point, 3).tolist()}")
    print(
        "上方目标 TCP m："
        f"({target_pose[0]:.6f}, {target_pose[1]:.6f}, {target_pose[2]:.6f})"
    )
    print(f"工具：tool={TOOL_INDEX}")
    print(f"等待时间：{WAIT_SECONDS:.1f} 秒")
    print(
        "动作：零位检查 -> 准备位 -> 开爪 -> 目标上方"
        " -> 等待 -> 空中闭爪 -> 抬高位"
        " -> 试管竖直位 -> 竖直后目标位"
        " -> 松爪下降位 -> 松开夹爪"
        " -> 松爪后目标位 -> 关节零位"
    )
    print(
        "闭爪后抬高关节 rad："
        f"{np.round(POST_CLOSE_LIFT_JOINTS, 6).tolist()}"
    )
    print(
        "试管竖直关节 rad："
        f"{np.round(TUBE_VERTICAL_JOINTS, 6).tolist()}"
    )
    print(
        "竖直后目标关节 rad："
        f"{np.round(POST_VERTICAL_TARGET_JOINTS, 6).tolist()}"
    )
    print(
        "松爪下降位关节 rad："
        f"{np.round(RELEASE_LOWER_JOINTS, 6).tolist()}"
    )
    print(
        "松爪后目标位关节 rad："
        f"{np.round(POST_RELEASE_TARGET_JOINTS, 6).tolist()}"
    )
    print("注意：图像角度暂时只显示，不直接作为机械臂 yaw。")
    print("最后会回到关节零位；不会宣称放置成功。")
    print("==========================================\n")


# =============================================================================
# 5. CArm 真机动作
# =============================================================================

def normalize_quaternion(quaternion):
    """检查并归一化 SDK 使用的 xyzw 四元数。"""

    quaternion = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if (
        quaternion.shape != (4,)
        or not np.isfinite(quaternion).all()
        or norm < 1e-9
    ):
        raise ValueError("四元数无效")
    return quaternion / norm


def tool_down_tilt_deg(quaternion):
    """计算工具 +Z 轴与机械臂基座 -Z 轴的夹角。"""

    qx, qy, _qz, _qw = normalize_quaternion(quaternion)
    base_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    return math.degrees(
        math.acos(float(np.clip(-base_z, -1.0, 1.0)))
    )


def quaternion_error_deg(actual, target):
    """计算两个四元数之间的最小姿态误差。"""

    dot = abs(
        float(
            np.dot(
                normalize_quaternion(actual),
                normalize_quaternion(target),
            )
        )
    )
    return math.degrees(
        2.0 * math.acos(float(np.clip(dot, 0.0, 1.0)))
    )


def accepted(response):
    """只有 SDK 明确返回 Task_Recieve 才算接受命令。"""

    return (
        isinstance(response, dict)
        and response.get("recv") == "Task_Recieve"
    )


def read_arm_state(arm):
    """读取并检查当前法兰 Pose 和六个关节角。"""

    pose = np.asarray(arm.cart_pose, dtype=float)
    joints = np.asarray(arm.joint_pos, dtype=float)
    if (
        pose.shape != (7,)
        or joints.shape != (6,)
        or not np.isfinite(pose).all()
        or not np.isfinite(joints).all()
    ):
        raise RuntimeError("无法读取完整的机械臂状态")
    return pose, joints


def create_error_monitor():
    """创建异步错误回调，以及主线程调用的错误检查函数。"""

    lock = threading.Lock()
    event = threading.Event()
    last_error = [None]

    def on_error(error_info):
        """记录 CArm SDK 异步回调报告的错误。"""

        try:
            if isinstance(error_info, dict):
                error_copy = dict(error_info)
            else:
                error_copy = {"error": None, "errMsg": str(error_info)}
        except Exception as exc:
            error_copy = {
                "error": None,
                "errMsg": f"错误回调异常：{exc}",
            }

        # 先保存错误再置位；本次真机流程中不再清除该错误。
        with lock:
            last_error[0] = error_copy
            event.set()

    def check(stage):
        """若已经收到异步错误，则立即抛出异常中止流程。"""

        if not event.is_set():
            return
        with lock:
            error_info = last_error[0]
        raise RuntimeError(
            f"{stage}：error={error_info.get('error')}，"
            f"{error_info.get('errMsg', '未知错误')}"
        )

    return on_error, check


def wait_until_connected(arm, timeout=3.0):
    """等待 SDK 完成连接。"""

    deadline = time.monotonic() + timeout
    while not arm.is_connected() and time.monotonic() < deadline:
        time.sleep(0.1)
    if not arm.is_connected():
        raise RuntimeError("机械臂连接失败")


def ensure_arm_idle(arm, check_error, stage):
    """发命令前确认连接、控制器状态和 tool=1。"""

    check_error(stage)
    if not arm.is_connected():
        raise RuntimeError(f"{stage}：机械臂连接断开")
    if arm.controller_state != 0:
        raise RuntimeError(
            f"{stage}：机械臂不是空闲状态，"
            f"controller_state={arm.controller_state}"
        )
    current_tool = int(arm.tool_index)
    if current_tool != TOOL_INDEX:
        raise RuntimeError(
            f"{stage}：当前 tool={current_tool}，"
            f"脚本要求 tool={TOOL_INDEX}"
        )


def wait_until_motion_stops(arm, check_error, stage):
    """同步命令返回后，再确认控制器已经稳定处于空闲状态。"""

    deadline = time.monotonic() + 2.0
    idle_since = None
    while time.monotonic() < deadline:
        check_error(stage)
        if not arm.is_connected():
            raise RuntimeError(f"{stage}：机械臂连接断开")

        state = arm.controller_state
        if state == -1:
            raise RuntimeError(f"{stage}：控制器进入错误状态")
        if state == 0:
            idle_since = idle_since or time.monotonic()
            if time.monotonic() - idle_since >= 0.2:
                check_error(stage)
                return
        else:
            idle_since = None
        time.sleep(0.05)

    raise RuntimeError(f"{stage}：等待机械臂停止超时")


def stop_safely(arm):
    """出现异常时尽力停止当前任务。"""

    try:
        arm.stop_task(at_once=True)
    except Exception as exc:
        print(f"警告：停止任务失败：{exc}")


def move_joint_checked(arm, joints, check_error, stage):
    """执行一次同步关节运动，并确认控制器没有报错。"""

    joint_array = np.asarray(joints, dtype=float)
    if joint_array.shape != (6,) or not np.isfinite(joint_array).all():
        raise RuntimeError(f"{stage}目标不是有效的六关节数组")

    ensure_arm_idle(arm, check_error, f"{stage}开始前")
    response = arm.move_joint(
        joint_array.tolist(),
        is_sync=True,
        tool=TOOL_INDEX,
    )
    if not accepted(response):
        raise RuntimeError(f"{stage}未被控制器接受：{response}")
    wait_until_motion_stops(arm, check_error, f"{stage}失败")
    return read_arm_state(arm)


def check_down_pose(pose, stage):
    """检查 TCP 是否足够高并接近垂直向下。"""

    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise RuntimeError(f"{stage}没有返回有效 Pose")

    quaternion = normalize_quaternion(pose[3:7])
    tilt_deg = tool_down_tilt_deg(quaternion)
    if pose[2] < MIN_READY_Z_M:
        raise RuntimeError(f"{stage}高度不足：Z={pose[2]:.3f} m")
    if tilt_deg > MAX_TILT_DEG:
        raise RuntimeError(f"{stage}没有朝下：倾角={tilt_deg:.2f}°")
    return quaternion, tilt_deg


def solve_target_joints(arm, target_pose, current_joints):
    """计算黄色试管上方目标的 IK，并限制关节跳变。"""

    response = arm.inverse_kine(
        target_pose,
        current_joints.tolist(),
        tool=TOOL_INDEX,
    )
    if not accepted(response):
        raise RuntimeError(f"目标 IK 求解失败：{response}")

    try:
        target_joints = np.asarray(
            response["data"]["joint1"],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("IK 没有返回有效关节解") from None

    if target_joints.shape != (6,) or not np.isfinite(target_joints).all():
        raise RuntimeError("IK 返回的关节解无效")

    max_change_deg = math.degrees(
        float(np.max(np.abs(target_joints - current_joints)))
    )
    if max_change_deg > 90.0:
        raise RuntimeError("IK 最大单关节变化超过 90°，拒绝运动")
    return target_joints, max_change_deg


def check_final_result(
    actual_pose,
    actual_joints,
    target_pose,
    target_joints,
):
    """检查黄色试管上方目标的位置、姿态和关节到位误差。"""

    position_error_mm = float(
        np.linalg.norm(actual_pose[:3] - target_pose[:3]) * 1000.0
    )
    orientation_error_deg = quaternion_error_deg(
        actual_pose[3:7],
        target_pose[3:7],
    )
    tilt_deg = tool_down_tilt_deg(actual_pose[3:7])
    joint_error_deg = math.degrees(
        float(np.max(np.abs(actual_joints - target_joints)))
    )

    print(f"最终位置误差: {position_error_mm:.3f} mm")
    print(f"最终姿态误差: {orientation_error_deg:.3f}°")
    print(f"最终向下倾角: {tilt_deg:.3f}°")
    print(f"最终最大关节误差: {joint_error_deg:.3f}°")

    if position_error_mm > MAX_POSITION_ERROR_MM:
        raise RuntimeError(
            f"最终位置误差过大：{position_error_mm:.3f} mm"
        )
    if orientation_error_deg > MAX_ORIENTATION_ERROR_DEG:
        raise RuntimeError(
            f"最终姿态误差过大：{orientation_error_deg:.3f}°"
        )
    if tilt_deg > MAX_TILT_DEG:
        raise RuntimeError(f"最终工具没有朝下：倾角={tilt_deg:.3f}°")
    if joint_error_deg > MAX_JOINT_ERROR_DEG:
        raise RuntimeError(
            f"最终关节误差过大：{joint_error_deg:.3f}°"
        )


def require_accepted(response, name) -> None:
    """SDK 没有明确接受命令时，立即终止后续动作。"""

    if not accepted(response):
        raise RuntimeError(f"{name}未被控制器接受：{response}")


def checked_tcp_distance(start_pose, end_pose, max_distance_m, name) -> float:
    """计算两个 TCP 的位置距离，超过本段上限时拒绝继续。"""

    start = np.asarray(start_pose, dtype=float).reshape(-1)
    end = np.asarray(end_pose, dtype=float).reshape(-1)
    if start.size < 3 or end.size < 3:
        raise ValueError(f"{name}缺少有效的 XYZ")
    if not np.isfinite(start[:3]).all() or not np.isfinite(end[:3]).all():
        raise ValueError(f"{name}包含非有限 XYZ")

    distance = float(np.linalg.norm(end[:3] - start[:3]))
    if distance > max_distance_m:
        raise RuntimeError(
            f"{name}距离过大：{distance:.3f} m，"
            f"上限 {max_distance_m:.3f} m"
        )
    return distance


def wait_for_joint_limits(arm, timeout=3.0) -> None:
    """等待 SDK 从控制器同步六个关节的上下限。"""

    deadline = time.monotonic() + timeout
    while not isinstance(getattr(arm, "limit", None), dict):
        if time.monotonic() >= deadline:
            raise RuntimeError("等待 CArm 关节限位超时")
        time.sleep(0.05)


def forward_kine_checked(arm, joints, stage) -> np.ndarray:
    """计算 tool=1 正运动学，并保留控制器的原始失败原因。

    CArm SDK 自带的 ``forward_kine`` 在控制器拒绝或超时时只返回 None，
    会丢失原始错误信息。这里直接发送同一个只读运动学请求，并把错误内容
    放进异常消息，便于判断是超时、拒绝还是关节越界。
    """

    joint_array = np.asarray(joints, dtype=float)
    if joint_array.shape != (6,) or not np.isfinite(joint_array).all():
        raise RuntimeError(f"{stage}关节角不是有效的 6 元数组")

    limits = getattr(arm, "limit", None)
    if not isinstance(limits, dict):
        raise RuntimeError(f"{stage}无法读取 CArm 关节限位")
    lower = np.asarray(limits.get("limit_lower", []), dtype=float)
    upper = np.asarray(limits.get("limit_upper", []), dtype=float)
    if (
        lower.shape != (6,)
        or upper.shape != (6,)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
    ):
        raise RuntimeError(f"{stage}的 CArm 关节限位无效")

    violations = np.flatnonzero(
        (joint_array < lower) | (joint_array > upper)
    )
    if violations.size:
        index = int(violations[0])
        raise RuntimeError(
            f"{stage}的 J{index + 1}={joint_array[index]:.6f} rad "
            f"超出 [{lower[index]:.6f}, {upper[index]:.6f}] rad"
        )

    last_response = None
    for attempt in range(1, 3):
        response = arm.request(
            {
                "command": "getKinematics",
                "task_id": "forward",
                "arm_index": arm.arm_index,
                "data": {
                    "tool": TOOL_INDEX,
                    "point_cnt": 1,
                    "joint1": joint_array.tolist(),
                },
            },
            timeout=3.0,
        )
        last_response = response

        if (
            isinstance(response, dict)
            and response.get("recv") == "Task_Recieve"
        ):
            try:
                pose = np.asarray(
                    response["data"]["point1"],
                    dtype=float,
                )
            except (KeyError, TypeError, ValueError):
                raise RuntimeError(
                    f"{stage} FK 响应缺少有效 point1：{response}"
                ) from None

            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise RuntimeError(
                    f"{stage} FK 返回的 Pose 无效：{response}"
                )
            return pose

        # 只读请求超时时允许再试一次；明确拒绝时不重复发送。
        error_text = ""
        if isinstance(response, dict):
            error_text = str(response.get("errMsg", ""))
        if "timed out" not in error_text.lower() or attempt == 2:
            break
        time.sleep(0.20)

    raise RuntimeError(f"{stage} FK 请求失败：{last_response}")


def check_horizontal_gripper_tcp(pose, stage, min_z_m) -> float:
    """检查 TCP 最低高度，以及夹爪是否接近水平。"""

    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise RuntimeError(f"{stage}没有返回有效 Pose")
    if pose[2] < min_z_m:
        raise RuntimeError(
            f"{stage}高度不足：Z={pose[2]:.3f} m，"
            f"要求至少 {min_z_m:.3f} m"
        )

    # 原有 check_down_pose 要求夹爪朝下；这里反过来要求夹爪接近水平。
    tilt_deg = tool_down_tilt_deg(pose[3:7])
    if not (
        MIN_VERTICAL_TOOL_TILT_DEG
        <= tilt_deg
        <= MAX_VERTICAL_TOOL_TILT_DEG
    ):
        raise RuntimeError(
            f"{stage}夹爪没有接近水平：相对向下方向 {tilt_deg:.2f}°"
        )
    return tilt_deg


def check_vertical_tcp(pose, stage) -> float:
    """检查试管竖直位 TCP。"""

    return check_horizontal_gripper_tcp(
        pose,
        stage,
        MIN_VERTICAL_TCP_Z_M,
    )


def check_post_vertical_tcp(pose, stage) -> float:
    """检查竖直后目标位 TCP。"""

    return check_horizontal_gripper_tcp(
        pose,
        stage,
        MIN_POST_VERTICAL_TCP_Z_M,
    )


def check_release_tcp(pose, stage) -> float:
    """检查下降后的松爪位 TCP。"""

    return check_horizontal_gripper_tcp(
        pose,
        stage,
        MIN_RELEASE_TCP_Z_M,
    )


def check_post_release_tcp(pose, stage) -> float:
    """检查松开夹爪以后到达的新目标位 TCP。"""

    return check_horizontal_gripper_tcp(
        pose,
        stage,
        MIN_POST_RELEASE_TCP_Z_M,
    )


def check_zero_tcp(pose, stage) -> None:
    """检查 tool=1 在关节零位时的 TCP 高度。"""

    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise RuntimeError(f"{stage}不是有效的 7 元 Pose")
    if pose[2] < MIN_ZERO_TCP_Z_M:
        raise RuntimeError(
            f"{stage}高度不足：z={pose[2]:.3f} m，"
            f"要求至少 {MIN_ZERO_TCP_Z_M:.3f} m"
        )


def check_motion_transition(
    start_tcp,
    start_joints,
    target_tcp,
    target_joints,
    name,
    max_distance_m=MAX_TRAVEL_M,
    max_joint_change_deg=90.0,
):
    """检查一段运动的 TCP 距离和最大单关节变化。"""

    distance = checked_tcp_distance(
        start_tcp,
        target_tcp,
        max_distance_m,
        name,
    )
    joint_change_deg = math.degrees(
        float(
            np.max(
                np.abs(
                    np.asarray(target_joints, dtype=float)
                    - np.asarray(start_joints, dtype=float)
                )
            )
        )
    )
    if joint_change_deg > max_joint_change_deg:
        raise RuntimeError(
            f"{name}的最大单关节变化超过 "
            f"{max_joint_change_deg:.1f}°，拒绝运动"
        )
    return distance, joint_change_deg


def check_fixed_target_result(
    name,
    actual_tcp,
    actual_joints,
    expected_tcp,
    expected_joints,
) -> None:
    """统一检查固定关节点位的位置、姿态和关节到位误差。"""

    position_error_mm = float(
        np.linalg.norm(actual_tcp[:3] - expected_tcp[:3]) * 1000.0
    )
    orientation_error_deg = quaternion_error_deg(
        actual_tcp[3:7],
        expected_tcp[3:7],
    )
    joint_error_deg = math.degrees(
        float(
            np.max(
                np.abs(
                    np.asarray(actual_joints, dtype=float)
                    - np.asarray(expected_joints, dtype=float)
                )
            )
        )
    )

    if position_error_mm > MAX_POSITION_ERROR_MM:
        raise RuntimeError(
            f"{name}位置误差过大：{position_error_mm:.3f} mm"
        )
    if orientation_error_deg > MAX_ORIENTATION_ERROR_DEG:
        raise RuntimeError(
            f"{name}姿态误差过大：{orientation_error_deg:.3f}°"
        )
    if joint_error_deg > MAX_JOINT_ERROR_DEG:
        raise RuntimeError(
            f"{name}关节误差过大：{joint_error_deg:.3f}°"
        )

    print(
        f"{name}误差：位置 {position_error_mm:.3f} mm，"
        f"姿态 {orientation_error_deg:.3f}°，"
        f"关节 {joint_error_deg:.3f}°"
    )


def precheck_fixed_target(
    arm,
    name,
    joints,
    pose_checker,
    reference_tcp=None,
):
    """在机械臂使能前，用只读 FK 检查一个固定关节点位。"""

    expected_tcp = forward_kine_checked(
        arm,
        joints,
        f"{name} tool=1",
    )
    pose_checker(expected_tcp, f"{name} tool=1 FK")

    message = (
        f"{name} FK TCP m："
        f"{np.round(expected_tcp[:3], 6).tolist()}"
    )
    if reference_tcp is not None:
        message += f"；网页截图参考：{reference_tcp.tolist()}"
    print(message)
    return expected_tcp


def move_to_fixed_target(
    arm,
    check_error,
    step_text,
    name,
    start_tcp,
    start_joints,
    target_joints,
    expected_tcp,
    pose_checker,
    *,
    max_distance_m=MAX_TRAVEL_M,
    max_joint_change_deg=90.0,
    checked_transition=None,
):
    """检查路径、执行固定关节运动，并统一验证实际到位结果。"""

    transition = checked_transition or check_motion_transition(
        start_tcp,
        start_joints,
        expected_tcp,
        target_joints,
        f"{name}路径",
        max_distance_m,
        max_joint_change_deg,
    )
    distance, joint_change_deg = transition
    print(
        f"{step_text}，TCP 距离 {distance:.3f} m，"
        f"最大关节变化 {joint_change_deg:.2f}°。"
    )

    _pose, actual_joints = move_joint_checked(
        arm,
        target_joints,
        check_error,
        f"{name}运动",
    )
    actual_tcp = forward_kine_checked(
        arm,
        actual_joints,
        f"实际{name} tool=1",
    )
    pose_checker(actual_tcp, f"实际{name} TCP")
    check_fixed_target_result(
        name,
        actual_tcp,
        actual_joints,
        expected_tcp,
        target_joints,
    )
    return actual_tcp, actual_joints


def set_gripper_checked(
    arm,
    check_error,
    step_text,
    action_name,
    opening_m,
) -> None:
    """确认机械臂空闲后发送夹爪命令，并检查异步错误。"""

    print(step_text)
    ensure_arm_idle(arm, check_error, f"{action_name}前")
    require_accepted(
        arm.set_gripper(opening_m, GRIPPER_FORCE_N),
        action_name,
    )
    time.sleep(1.0)
    check_error(f"{action_name}失败")


def execute_demo(target_pose, ip) -> None:
    """执行抓取、松爪、松爪后运动，最后返回关节零位。"""

    confirmation_text = (
        f"GRASP_PLACE_RELEASE_RETURN_ZERO tool=1 "
        f"x={target_pose[0]:.3f} "
        f"y={target_pose[1]:.3f} "
        f"z={target_pose[2]:.3f} wait=3"
    )
    print("请清空运动路径、确认夹爪中没有物体，并准备好急停。")
    typed = input(f"请输入完整确认文字：\n{confirmation_text}\n> ").strip()
    if typed != confirmation_text:
        print("确认文字不匹配：没有连接机械臂，也没有运动。")
        return

    # 延迟导入：默认 dry-run 不会加载或连接 CArm。
    from carm import Carm

    arm = None
    try:
        arm = Carm(addr=ip)
        wait_until_connected(arm)

        # “从零位开始”采用只读检查，不额外发送一次盲目回零命令。
        _pose, joints = read_arm_state(arm)
        zero_error_deg = math.degrees(float(np.max(np.abs(joints))))
        print(f"起始零位最大关节误差：{zero_error_deg:.3f}°")
        if zero_error_deg > MAX_JOINT_ERROR_DEG:
            raise RuntimeError("机械臂当前不在关节零位，拒绝开始")

        # 等待 SDK 同步真实关节限位，再检查全部固定目标。
        # 这些都是只读 FK，全部通过后才允许机械臂使能。
        wait_for_joint_limits(arm)

        # 先只读检查所有固定关节点位。重复的 FK 和打印由同一函数完成。
        ready_fk = precheck_fixed_target(
            arm,
            "准备位",
            DOWN_READY_JOINTS,
            check_down_pose,
        )
        lift_fk = precheck_fixed_target(
            arm,
            "闭爪后抬高位",
            POST_CLOSE_LIFT_JOINTS,
            check_down_pose,
            POST_CLOSE_REFERENCE_TCP_M,
        )

        # 先使用视觉目标和固定抬高位 FK 做计划距离检查。
        # 该检查发生在 set_ready 之前，失败时机械臂完全不会使能。
        target = np.asarray(target_pose, dtype=float)
        planned_lift_distance = checked_tcp_distance(
            target,
            lift_fk,
            MAX_TARGET_TO_LIFT_M,
            "计划目标上方到抬高位",
        )
        print(
            "目标上方到抬高位预检距离："
            f"{planned_lift_distance:.3f} m；"
            f"本段上限：{MAX_TARGET_TO_LIFT_M:.3f} m"
        )

        # 竖直位的夹爪应接近水平，不能使用上面的“朝下”检查。
        vertical_fk = precheck_fixed_target(
            arm,
            "试管竖直位",
            TUBE_VERTICAL_JOINTS,
            check_vertical_tcp,
            TUBE_VERTICAL_REFERENCE_TCP_M,
        )
        post_vertical_fk = precheck_fixed_target(
            arm,
            "竖直后目标位",
            POST_VERTICAL_TARGET_JOINTS,
            check_post_vertical_tcp,
            POST_VERTICAL_REFERENCE_TCP_M,
        )
        release_fk = precheck_fixed_target(
            arm,
            "松爪下降位",
            RELEASE_LOWER_JOINTS,
            check_release_tcp,
            RELEASE_LOWER_REFERENCE_TCP_M,
        )
        post_release_fk = precheck_fixed_target(
            arm,
            "松爪后目标位",
            POST_RELEASE_TARGET_JOINTS,
            check_post_release_tcp,
            POST_RELEASE_REFERENCE_TCP_M,
        )
        zero_fk = precheck_fixed_target(
            arm,
            "关节零位",
            ZERO_JOINTS,
            check_zero_tcp,
        )

        # 这些固定目标之间的路径也能在使能前一次性完成预检。
        planned_release = check_motion_transition(
            post_vertical_fk,
            POST_VERTICAL_TARGET_JOINTS,
            release_fk,
            RELEASE_LOWER_JOINTS,
            "计划竖直后目标位到松爪下降位",
            MAX_TRAVEL_M,
            90.0,
        )
        print(
            "松爪下降段预检："
            f"TCP 距离 {planned_release[0]:.3f} m，"
            f"最大关节变化 {planned_release[1]:.2f}°"
        )
        planned_post_release = check_motion_transition(
            release_fk,
            RELEASE_LOWER_JOINTS,
            post_release_fk,
            POST_RELEASE_TARGET_JOINTS,
            "计划松爪下降位到松爪后目标位",
            MAX_TRAVEL_M,
            90.0,
        )
        print(
            "松爪后运动预检："
            f"TCP 距离 {planned_post_release[0]:.3f} m，"
            f"最大关节变化 {planned_post_release[1]:.2f}°"
        )
        planned_return_zero = check_motion_transition(
            post_release_fk,
            POST_RELEASE_TARGET_JOINTS,
            zero_fk,
            ZERO_JOINTS,
            "计划松爪后目标位到关节零位",
            MAX_TRAVEL_M,
            MAX_RETURN_ZERO_JOINT_CHANGE_DEG,
        )
        print(
            "回零运动预检："
            f"TCP 距离 {planned_return_zero[0]:.3f} m，"
            f"最大关节变化 {planned_return_zero[1]:.2f}°"
        )

        if not arm.set_ready():
            raise RuntimeError("机械臂无法进入就绪状态")

        on_error, check_error = create_error_monitor()
        arm.on_error(on_error)

        require_accepted(arm.set_tool_index(TOOL_INDEX), "设置 tool=1")
        require_accepted(arm.set_collision_config(True, 2), "设置碰撞检测")
        require_accepted(arm.set_speed_level(1.8, 80), "设置速度")

        # 等待控制器状态真实更新成 tool=1。
        deadline = time.monotonic() + 2.0
        while int(arm.tool_index) != TOOL_INDEX and time.monotonic() < deadline:
            time.sleep(0.05)
        if int(arm.tool_index) != TOOL_INDEX:
            raise RuntimeError(f"控制器当前 tool={arm.tool_index}，不是 tool=1")

        print("1/11：从零位移动到垂直向下准备位。")
        _pose, ready_joints = move_joint_checked(
            arm,
            DOWN_READY_JOINTS,
            check_error,
            "准备位运动",
        )
        ready_tcp = forward_kine_checked(
            arm,
            ready_joints,
            "实际准备位 tool=1",
        )
        check_down_pose(ready_tcp, "实际准备位 TCP")

        set_gripper_checked(
            arm,
            check_error,
            "2/11：在准备位张开夹爪。",
            "张开夹爪",
            GRIPPER_OPEN_M,
        )

        distance = float(np.linalg.norm(target[:3] - ready_tcp[:3]))
        if distance > MAX_TRAVEL_M:
            raise RuntimeError(f"准备位到目标距离过大：{distance:.3f} m")

        _pose, current_joints = read_arm_state(arm)
        target_joints, max_change_deg = solve_target_joints(
            arm,
            target.tolist(),
            current_joints,
        )
        print(
            f"3/11：移动到黄色试管中心上方，距离 {distance:.3f} m，"
            f"最大关节变化 {max_change_deg:.2f}°。"
        )
        _pose, actual_joints = move_joint_checked(
            arm,
            target_joints,
            check_error,
            "目标上方运动",
        )

        # SDK cart_pose 是法兰；用 tool=1 正运动学得到实际 TCP 后再检查。
        actual_tcp = forward_kine_checked(
            arm,
            actual_joints,
            "实际目标上方 tool=1",
        )
        check_final_result(
            actual_tcp,
            actual_joints,
            target,
            target_joints,
        )

        # 使用真实到位结果再次检查下一段运动。
        # 必须在等待和闭爪之前通过，避免夹爪闭合后才发现不能抬高。
        lift_transition = check_motion_transition(
            actual_tcp,
            actual_joints,
            lift_fk,
            POST_CLOSE_LIFT_JOINTS,
            "实际目标上方到抬高位",
            MAX_TARGET_TO_LIFT_M,
            90.0,
        )
        print(
            "闭爪前抬高路径复检通过："
            f"TCP 距离 {lift_transition[0]:.3f} m，"
            f"最大关节变化 {lift_transition[1]:.2f}°。"
        )

        print(f"已经到达目标上方，等待 {WAIT_SECONDS:.1f} 秒……")
        time.sleep(WAIT_SECONDS)
        check_error("等待期间机械臂报错")

        set_gripper_checked(
            arm,
            check_error,
            "4/11：在目标上方闭合夹爪。",
            "闭合夹爪",
            GRIPPER_CLOSE_M,
        )

        # 第 5～8 步都是“固定关节点位运动”，统一执行相同的安全流程。
        lift_actual_tcp, lift_actual_joints = move_to_fixed_target(
            arm,
            check_error,
            "5/11：闭爪后移动到抬高位",
            "闭爪后抬高位",
            actual_tcp,
            actual_joints,
            POST_CLOSE_LIFT_JOINTS,
            lift_fk,
            check_down_pose,
            max_distance_m=MAX_TARGET_TO_LIFT_M,
            max_joint_change_deg=90.0,
            checked_transition=lift_transition,
        )
        vertical_actual_tcp, vertical_actual_joints = move_to_fixed_target(
            arm,
            check_error,
            "6/11：直接移动到试管竖直位",
            "试管竖直位",
            lift_actual_tcp,
            lift_actual_joints,
            TUBE_VERTICAL_JOINTS,
            vertical_fk,
            check_vertical_tcp,
            max_distance_m=MAX_TRAVEL_M,
            max_joint_change_deg=MAX_DIRECT_VERTICAL_JOINT_CHANGE_DEG,
        )
        (
            post_vertical_actual_tcp,
            post_vertical_actual_joints,
        ) = move_to_fixed_target(
            arm,
            check_error,
            "7/11：移动到竖直后目标位",
            "竖直后目标位",
            vertical_actual_tcp,
            vertical_actual_joints,
            POST_VERTICAL_TARGET_JOINTS,
            post_vertical_fk,
            check_post_vertical_tcp,
        )
        release_actual_tcp, release_actual_joints = move_to_fixed_target(
            arm,
            check_error,
            "8/11：下降到松爪位",
            "松爪下降位",
            post_vertical_actual_tcp,
            post_vertical_actual_joints,
            RELEASE_LOWER_JOINTS,
            release_fk,
            check_release_tcp,
        )

        # 先用下降位的实际结果复检松爪后的下一段运动。
        # 如果下一段不安全，就不先松爪，避免松爪后才发现无法继续。
        post_release_transition = check_motion_transition(
            release_actual_tcp,
            release_actual_joints,
            post_release_fk,
            POST_RELEASE_TARGET_JOINTS,
            "实际松爪下降位到松爪后目标位",
            MAX_TRAVEL_M,
            90.0,
        )

        # 第 9 步：只有下降位和下一段路径都通过检查，才允许松开夹爪。
        set_gripper_checked(
            arm,
            check_error,
            "9/11：已经到达松爪下降位，松开夹爪。",
            "松开夹爪",
            GRIPPER_OPEN_M,
        )

        # 第 10 步：松爪完成以后，移动到用户最新截图中的固定点位。
        (
            post_release_actual_tcp,
            post_release_actual_joints,
        ) = move_to_fixed_target(
            arm,
            check_error,
            "10/11：移动到松爪后目标位",
            "松爪后目标位",
            release_actual_tcp,
            release_actual_joints,
            POST_RELEASE_TARGET_JOINTS,
            post_release_fk,
            check_post_release_tcp,
            checked_transition=post_release_transition,
        )

        # 第 11 步：使用实际到位结果复检后，再返回六关节零位。
        move_to_fixed_target(
            arm,
            check_error,
            "11/11：返回关节零位",
            "最终零位",
            post_release_actual_tcp,
            post_release_actual_joints,
            ZERO_JOINTS,
            zero_fk,
            check_zero_tcp,
            max_distance_m=MAX_TRAVEL_M,
            max_joint_change_deg=MAX_RETURN_ZERO_JOINT_CHANGE_DEG,
        )

        print(
            "测试完成：机械臂已经松开夹爪，并回到关节零位。"
        )
    except KeyboardInterrupt:
        print("\n用户中断，正在停止任务。")
        if arm is not None:
            stop_safely(arm)
        raise
    except Exception:
        if arm is not None:
            stop_safely(arm)
        raise
    finally:
        if arm is not None:
            arm.disconnect()


# =============================================================================
# 6. 命令行入口
# =============================================================================

def parse_args():
    """读取终端参数，并返回 argparse.Namespace。"""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        default=None,
        help="YOLO Seg 权重路径；省略时寻找 runs 下最新的 best.pt。",
    )
    parser.add_argument(
        "--serial",
        default=None,
        help="RealSense 序列号；连接多台相机时必须指定。",
    )
    parser.add_argument(
        "--pixel",
        nargs=2,
        type=float,
        metavar=("U", "V"),
        help="跳过 RealSense，直接使用已确认的黄色中心像素做 dry-run。",
    )
    parser.add_argument(
        "--ip",
        default="10.42.0.101",
        help="CArm 控制器 IP，默认 10.42.0.101。",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="连接 CArm 执行动作；省略时只打印计划。",
    )
    return parser.parse_args()


def main() -> int:
    """按照本脚本的编号流程依次执行各个步骤。"""

    args = parse_args()

    # 固定像素只用于离线复算。真机必须重新读取相机并锁定当前稳定中心，
    # 避免因为试管或相机已经移动，仍拿旧像素驱动机械臂。
    if args.execute and args.pixel is not None:
        raise ValueError("--pixel 只能用于 dry-run，不能和 --execute 一起使用")

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

    # --pixel 用于重复检查一个已知中心；省略时才打开 RealSense。
    if args.pixel is None:
        center, angle_deg = run_preview(args)
    else:
        center = (float(args.pixel[0]), float(args.pixel[1]))
        angle_deg = None

    target_pose, workspace_point, arm_point, _lift = pixel_to_target(center)
    print_plan(
        center,
        angle_deg,
        target_pose,
        workspace_point,
        arm_point,
    )

    if args.execute:
        execute_demo(target_pose, args.ip)
    else:
        print("[DRY-RUN] 没有连接机械臂，也没有发送夹爪命令。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("\n用户取消，相机窗口已经关闭。") from None
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"程序终止：{exc}") from None
