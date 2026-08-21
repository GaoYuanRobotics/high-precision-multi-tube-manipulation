#!/usr/bin/env python3
"""在安全高位测试夹爪 yaw，不下降、不闭爪，观察后自动回零。

先运行脚本 11，记录它输出的机械臂平面中心 x/y 和基座 B→C yaw，再把这三个
数值传给本脚本。本脚本会生成相差 180° 的两个姿态，连接 CArm 后分别求 IK，
选择最大单关节变化较小的候选。

默认只打印计划。只有显式添加 ``--execute``、通过两次完整文字确认后才会：

    零位检查 -> 准备位 -> 张开夹爪 -> 安全高位 yaw 对齐
    -> 等待人工观察 -> 准备位 -> 关节零位

本脚本没有 RealSense、YOLO、下降动作或闭爪动作，也不调用其他编号脚本。
"""

from __future__ import annotations

import argparse
import math
import threading
import time

import numpy as np


# =============================================================================
# 1. 当前机械臂和安全高位参数
# =============================================================================

TOOL_INDEX = 1
DEFAULT_IP = "10.42.0.101"

# 与脚本 10/11 当前使用值一致的“夹爪向下、参考 yaw=0”四元数。
DOWN_QUATERNION = np.array(
    [0.999575504, 0.008135427, 0.027844000, 0.002709061],
    dtype=float,
)

ZERO_JOINTS = np.zeros(6, dtype=float)
DOWN_READY_JOINTS = np.array(
    [-0.001726, 1.751210, -0.626573, -0.000954, 0.446518, -0.000954],
    dtype=float,
)

DEFAULT_ALIGN_Z_M = 0.300
MIN_ALIGN_Z_M = 0.250
MAX_ALIGN_Z_M = 0.450
MAX_TCP_STEP_M = 0.300
MAX_IK_JOINT_CHANGE_DEG = 95.0

MAX_START_ZERO_ERROR_DEG = 3.0
MAX_POSITION_ERROR_MM = 2.0
MAX_ORIENTATION_ERROR_DEG = 1.0
MAX_JOINT_ERROR_DEG = 3.0
MAX_DOWN_TILT_DEG = 10.0

GRIPPER_OPEN_M = 0.060
GRIPPER_FORCE_N = 10.0


# =============================================================================
# 2. yaw 和四元数
# =============================================================================

def wrap_pi(angle_rad: float) -> float:
    """把角度限制到 [-pi, pi)。"""

    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def normalize_quaternion(quaternion) -> np.ndarray:
    """检查并归一化 [qx, qy, qz, qw]。"""

    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("四元数必须是四个有限数字")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("四元数长度不能为 0")
    return q / norm


def quaternion_multiply(left, right) -> np.ndarray:
    """计算 [qx,qy,qz,qw] Hamilton 乘积。"""

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


def quaternion_error_deg(first, second) -> float:
    """计算两个四元数之间的最小夹角。"""

    dot = abs(
        float(
            np.dot(
                normalize_quaternion(first),
                normalize_quaternion(second),
            )
        )
    )
    return math.degrees(
        2.0 * math.acos(float(np.clip(dot, 0.0, 1.0)))
    )


def tool_down_tilt_deg(quaternion) -> float:
    """计算工具 +Z 轴和机械臂基座 -Z 轴之间的夹角。"""

    qx, qy, _qz, _qw = normalize_quaternion(quaternion)
    tool_z_in_base_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    return math.degrees(
        math.acos(float(np.clip(-tool_z_in_base_z, -1.0, 1.0)))
    )


def make_candidates(
    x_m: float,
    y_m: float,
    align_z_m: float,
    tube_yaw_deg: float,
    yaw_offset_deg: float,
) -> list[dict]:
    """生成两个相差 180° 的安全高位候选 Pose。"""

    first_yaw = wrap_pi(
        math.radians(tube_yaw_deg + yaw_offset_deg)
    )
    yaws = [first_yaw, wrap_pi(first_yaw + math.pi)]

    candidates = []
    for yaw_index in range(len(yaws)):
        index = yaw_index + 1
        yaw_rad = yaws[yaw_index]
        quaternion = quaternion_multiply(
            yaw_quaternion(yaw_rad),
            DOWN_QUATERNION,
        )
        candidates.append(
            {
                "index": index,
                "yaw_rad": yaw_rad,
                "pose": np.array(
                    [
                        x_m,
                        y_m,
                        align_z_m,
                        quaternion[0],
                        quaternion[1],
                        quaternion[2],
                        quaternion[3],
                    ],
                    dtype=float,
                ),
            }
        )
    return candidates


# =============================================================================
# 3. CArm 状态、FK/IK 和运动检查
# =============================================================================

def accepted(response) -> bool:
    """只有 SDK 明确返回 Task_Recieve 才算接受。"""

    return (
        isinstance(response, dict)
        and response.get("recv") == "Task_Recieve"
    )


def require_accepted(response, action: str) -> None:
    """确认 CArm SDK 已接受命令，否则抛出异常。"""

    if not accepted(response):
        raise RuntimeError(f"{action}未被控制器接受：{response}")


def read_arm_state(arm):
    """读取法兰 Pose 和六个关节角，并检查数据完整。"""

    pose = np.asarray(arm.cart_pose, dtype=float)
    joints = np.asarray(arm.joint_pos, dtype=float)
    if (
        pose.shape != (7,)
        or joints.shape != (6,)
        or not np.isfinite(pose).all()
        or not np.isfinite(joints).all()
    ):
        raise RuntimeError("无法读取完整机械臂状态")
    return pose, joints


def wait_until_connected(arm, timeout_s: float = 3.0) -> None:
    """在超时范围内等待 CArm WebSocket 建立连接。"""

    deadline = time.monotonic() + timeout_s
    while not arm.is_connected() and time.monotonic() < deadline:
        time.sleep(0.1)
    if not arm.is_connected():
        raise RuntimeError("机械臂连接失败")


def wait_for_joint_limits(arm, timeout_s: float = 3.0) -> None:
    """等待控制器同步真实关节限位。"""

    deadline = time.monotonic() + timeout_s
    while not isinstance(getattr(arm, "limit", None), dict):
        if time.monotonic() >= deadline:
            raise RuntimeError("等待 CArm 关节限位超时")
        time.sleep(0.05)


def check_joint_limits(arm, joints, stage: str) -> np.ndarray:
    """检查六关节数组和控制器真实上下限。"""

    joint_array = np.asarray(joints, dtype=float)
    if joint_array.shape != (6,) or not np.isfinite(joint_array).all():
        raise RuntimeError(f"{stage}不是有效的六关节数组")

    limits = getattr(arm, "limit", None)
    if not isinstance(limits, dict):
        raise RuntimeError(f"{stage}无法读取关节限位")
    lower = np.asarray(limits.get("limit_lower", []), dtype=float)
    upper = np.asarray(limits.get("limit_upper", []), dtype=float)
    if lower.shape != (6,) or upper.shape != (6,):
        raise RuntimeError(f"{stage}关节限位格式无效")

    violations = np.flatnonzero(
        (joint_array < lower) | (joint_array > upper)
    )
    if violations.size:
        index = int(violations[0])
        raise RuntimeError(
            f"{stage} J{index + 1}={joint_array[index]:.6f} rad "
            f"超出 [{lower[index]:.6f}, {upper[index]:.6f}]"
        )
    return joint_array


def forward_kine_checked(arm, joints, stage: str) -> np.ndarray:
    """计算 tool=1 正运动学，并保留控制器原始错误。"""

    joint_array = check_joint_limits(arm, joints, stage)
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
    if not accepted(response):
        raise RuntimeError(f"{stage} FK 请求失败：{response}")
    try:
        pose = np.asarray(response["data"]["point1"], dtype=float)
    except (KeyError, TypeError, ValueError):
        raise RuntimeError(f"{stage} FK 响应无有效 point1：{response}") from None
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise RuntimeError(f"{stage} FK Pose 无效")
    return pose


def solve_candidate(arm, candidate, reference_joints) -> dict | None:
    """求一个候选的 IK；失败或跳变过大时返回 None。"""

    response = arm.inverse_kine(
        candidate["pose"].tolist(),
        np.asarray(reference_joints, dtype=float).tolist(),
        tool=TOOL_INDEX,
    )
    if not accepted(response):
        print(f"候选 {candidate['index']} IK 被拒绝：{response}")
        return None
    try:
        joints = np.asarray(response["data"]["joint1"], dtype=float)
    except (KeyError, TypeError, ValueError):
        print(f"候选 {candidate['index']} IK 没有有效关节解")
        return None

    try:
        joints = check_joint_limits(
            arm,
            joints,
            f"候选 {candidate['index']}",
        )
        fk_pose = forward_kine_checked(
            arm,
            joints,
            f"候选 {candidate['index']}",
        )
    except RuntimeError as exc:
        print(f"候选 {candidate['index']} 无效：{exc}")
        return None

    max_change_deg = math.degrees(
        float(np.max(np.abs(joints - reference_joints)))
    )
    position_error_mm = float(
        np.linalg.norm(fk_pose[:3] - candidate["pose"][:3]) * 1000.0
    )
    orientation_error_deg = quaternion_error_deg(
        fk_pose[3:7],
        candidate["pose"][3:7],
    )
    tilt_deg = tool_down_tilt_deg(fk_pose[3:7])
    if max_change_deg > MAX_IK_JOINT_CHANGE_DEG:
        print(
            f"候选 {candidate['index']} 最大关节变化 "
            f"{max_change_deg:.2f}°，超过限制"
        )
        return None
    if (
        position_error_mm > MAX_POSITION_ERROR_MM
        or orientation_error_deg > MAX_ORIENTATION_ERROR_DEG
        or tilt_deg > MAX_DOWN_TILT_DEG
    ):
        print(
            f"候选 {candidate['index']} FK 验证失败："
            f"位置 {position_error_mm:.3f} mm，"
            f"姿态 {orientation_error_deg:.3f}°，"
            f"倾角 {tilt_deg:.3f}°"
        )
        return None

    result = dict(candidate)
    result.update(
        {
            "joints": joints,
            "fk_pose": fk_pose,
            "max_change_deg": max_change_deg,
        }
    )
    return result


def create_error_monitor():
    """把 SDK 异步错误安全地传回主线程。"""

    lock = threading.Lock()
    event = threading.Event()
    last_error = [None]

    def on_error(error_info):
        """记录 CArm SDK 异步回调报告的错误。"""

        if isinstance(error_info, dict):
            error_copy = dict(error_info)
        else:
            error_copy = {"error": None, "errMsg": str(error_info)}
        with lock:
            last_error[0] = error_copy
            event.set()

    def check(stage: str):
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


def ensure_idle(arm, check_error, stage: str) -> None:
    """确认机械臂当前没有仍在执行的运动任务。"""

    check_error(stage)
    if not arm.is_connected():
        raise RuntimeError(f"{stage}：机械臂连接断开")
    if arm.controller_state != 0:
        raise RuntimeError(
            f"{stage}：controller_state={arm.controller_state}，不是空闲"
        )
    if int(arm.tool_index) != TOOL_INDEX:
        raise RuntimeError(
            f"{stage}：当前 tool={arm.tool_index}，要求 tool={TOOL_INDEX}"
        )


def wait_until_stopped(arm, check_error, stage: str) -> None:
    """同步运动返回后，继续确认控制器稳定空闲。"""

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
                return
        else:
            idle_since = None
        time.sleep(0.05)
    raise RuntimeError(f"{stage}：等待停止超时")


def move_joint_checked(arm, joints, check_error, stage: str):
    """执行一次同步关节运动。"""

    joint_array = check_joint_limits(arm, joints, stage)
    ensure_idle(arm, check_error, f"{stage}开始前")
    response = arm.move_joint(
        joint_array.tolist(),
        is_sync=True,
        tool=TOOL_INDEX,
    )
    require_accepted(response, stage)
    wait_until_stopped(arm, check_error, f"{stage}失败")
    return read_arm_state(arm)


def check_arrival(
    arm,
    actual_joints,
    expected_joints,
    expected_pose,
    stage: str,
    *,
    require_down: bool = True,
) -> np.ndarray:
    """用 tool=1 FK 检查位置、姿态和关节到位误差。"""

    actual_pose = forward_kine_checked(arm, actual_joints, stage)
    position_error_mm = float(
        np.linalg.norm(actual_pose[:3] - expected_pose[:3]) * 1000.0
    )
    orientation_error_deg = quaternion_error_deg(
        actual_pose[3:7],
        expected_pose[3:7],
    )
    joint_error_deg = math.degrees(
        float(np.max(np.abs(actual_joints - expected_joints)))
    )
    tilt_deg = tool_down_tilt_deg(actual_pose[3:7])
    print(
        f"{stage}误差：位置 {position_error_mm:.3f} mm，"
        f"姿态 {orientation_error_deg:.3f}°，"
        f"关节 {joint_error_deg:.3f}°，"
        f"向下倾角 {tilt_deg:.3f}°"
    )
    if position_error_mm > MAX_POSITION_ERROR_MM:
        raise RuntimeError(f"{stage}位置误差过大")
    if orientation_error_deg > MAX_ORIENTATION_ERROR_DEG:
        raise RuntimeError(f"{stage}姿态误差过大")
    if joint_error_deg > MAX_JOINT_ERROR_DEG:
        raise RuntimeError(f"{stage}关节误差过大")
    if require_down and tilt_deg > MAX_DOWN_TILT_DEG:
        raise RuntimeError(f"{stage}工具没有保持向下")
    return actual_pose


def tcp_distance(start_pose, end_pose, stage: str) -> float:
    """限制一次运动两端 TCP 的直线距离。"""

    distance = float(
        np.linalg.norm(
            np.asarray(end_pose, dtype=float)[:3]
            - np.asarray(start_pose, dtype=float)[:3]
        )
    )
    if distance > MAX_TCP_STEP_M:
        raise RuntimeError(
            f"{stage} TCP 距离 {distance:.3f} m，"
            f"超过 {MAX_TCP_STEP_M:.3f} m"
        )
    return distance


def stop_safely(arm) -> None:
    """请求停止机械臂，并保留原始错误供用户判断。"""

    try:
        arm.stop_task(at_once=True)
    except Exception as exc:
        print(f"警告：停止任务失败：{exc}")


# =============================================================================
# 4. 高空对齐真机流程
# =============================================================================

def print_plan(args, candidates) -> None:
    """把即将检查或执行的机械臂计划打印到终端。"""

    print("\n========== 夹爪高空 yaw 对齐计划 ==========")
    print(f"黄色中心基座 XY：({args.x:.6f}, {args.y:.6f}) m")
    print(f"安全对齐高度：{args.align_z:.3f} m")
    print(f"试管基座 B→C yaw：{args.tube_yaw_deg:.3f}°")
    print(f"夹爪 yaw 偏移：{args.yaw_offset_deg:.3f}°")
    for candidate in candidates:
        print(
            f"候选 {candidate['index']} yaw："
            f"{math.degrees(candidate['yaw_rad']):.3f}°"
        )
        print(
            "  Pose："
            f"{np.round(candidate['pose'], 9).tolist()}"
        )
    print("动作：零位 -> 准备位 -> 开爪 -> 高空对齐 -> 观察 -> 准备位 -> 零位")
    print("不会下降，不会闭爪，不会宣称抓取成功。")
    print("============================================\n")


def execute_alignment(args, candidates) -> None:
    """连接 CArm，自动选择 IK 更近的候选并执行一次高空观察。"""

    first_confirmation = (
        f"YAW_ALIGNMENT_READONLY tool=1 "
        f"x={args.x:.3f} y={args.y:.3f} z={args.align_z:.3f}"
    )
    print("请清空机械臂周围路径、准备急停，并确认夹爪中没有物体。")
    typed = input(
        f"首先只连接并计算 IK，请输入：\n{first_confirmation}\n> "
    ).strip()
    if typed != first_confirmation:
        print("确认文字不匹配：没有连接机械臂。")
        return

    # 默认 dry-run 不导入 CArm；只有完成第一次确认才加载 SDK。
    from carm import Carm

    arm = None
    try:
        arm = Carm(addr=args.ip)
        wait_until_connected(arm)

        _flange_pose, start_joints = read_arm_state(arm)
        start_error_deg = math.degrees(
            float(np.max(np.abs(start_joints - ZERO_JOINTS)))
        )
        print(f"起始零位最大关节误差：{start_error_deg:.3f}°")
        if start_error_deg > MAX_START_ZERO_ERROR_DEG:
            raise RuntimeError("机械臂当前不在关节零位，拒绝开始")

        wait_for_joint_limits(arm)
        zero_pose = forward_kine_checked(arm, ZERO_JOINTS, "零位")
        ready_pose = forward_kine_checked(
            arm,
            DOWN_READY_JOINTS,
            "准备位",
        )

        valid_candidates = []
        for candidate in candidates:
            solved = solve_candidate(
                arm,
                candidate,
                DOWN_READY_JOINTS,
            )
            if solved is not None:
                valid_candidates.append(solved)
        if not valid_candidates:
            raise RuntimeError("两个 yaw 候选都没有通过 IK/FK 安全检查")
        selected = valid_candidates[0]
        for candidate in valid_candidates[1:]:
            if candidate["max_change_deg"] < selected["max_change_deg"]:
                selected = candidate

        ready_to_target_m = tcp_distance(
            ready_pose,
            selected["fk_pose"],
            "准备位到高空对齐位",
        )
        tcp_distance(
            selected["fk_pose"],
            ready_pose,
            "高空对齐位返回准备位",
        )
        tcp_distance(ready_pose, zero_pose, "准备位返回零位")

        print(
            f"自动选择候选 {selected['index']}："
            f"yaw={math.degrees(selected['yaw_rad']):.3f}°，"
            f"最大关节变化={selected['max_change_deg']:.2f}°，"
            f"TCP 距离={ready_to_target_m:.3f} m"
        )
        print(f"目标关节 rad：{np.round(selected['joints'], 6).tolist()}")

        final_confirmation = (
            f"MOVE_HIGH_YAW candidate={selected['index']} tool=1 "
            f"yaw={math.degrees(selected['yaw_rad']):.3f} "
            f"z={args.align_z:.3f}"
        )
        typed = input(
            "只会张开夹爪并移动到高空对齐位；"
            f"请输入：\n{final_confirmation}\n> "
        ).strip()
        if typed != final_confirmation:
            print("第二次确认不匹配：没有使能，也没有运动。")
            return

        if not arm.set_ready():
            raise RuntimeError("机械臂无法进入就绪状态")
        on_error, check_error = create_error_monitor()
        arm.on_error(on_error)

        require_accepted(arm.set_tool_index(TOOL_INDEX), "设置 tool=1")
        require_accepted(
            arm.set_collision_config(True, 2),
            "设置碰撞检测",
        )
        require_accepted(arm.set_speed_level(1.0, 80), "设置低速")

        deadline = time.monotonic() + 2.0
        while int(arm.tool_index) != TOOL_INDEX and time.monotonic() < deadline:
            time.sleep(0.05)
        if int(arm.tool_index) != TOOL_INDEX:
            raise RuntimeError("控制器没有切换到 tool=1")

        print("1/5：零位移动到垂直向下准备位。")
        _pose, ready_actual_joints = move_joint_checked(
            arm,
            DOWN_READY_JOINTS,
            check_error,
            "准备位运动",
        )
        check_arrival(
            arm,
            ready_actual_joints,
            DOWN_READY_JOINTS,
            ready_pose,
            "准备位",
        )

        print("2/5：在准备位张开夹爪。")
        ensure_idle(arm, check_error, "张开夹爪前")
        require_accepted(
            arm.set_gripper(GRIPPER_OPEN_M, GRIPPER_FORCE_N),
            "张开夹爪",
        )
        time.sleep(1.0)
        check_error("张开夹爪失败")

        print(
            f"3/5：移动到高空对齐位，候选 {selected['index']}，"
            f"yaw={math.degrees(selected['yaw_rad']):.3f}°。"
        )
        _pose, target_actual_joints = move_joint_checked(
            arm,
            selected["joints"],
            check_error,
            "高空 yaw 对齐运动",
        )
        check_arrival(
            arm,
            target_actual_joints,
            selected["joints"],
            selected["pose"],
            "高空 yaw 对齐位",
        )

        print("\n请从上方观察：两个手指是否分别位于试管两侧。")
        print("此时夹爪保持张开，不会下降，也不会闭合。")
        if input("观察完成后请输入 RETURN 返回准备位和零位：\n> ").strip() != "RETURN":
            raise RuntimeError("没有收到 RETURN，立即停止当前任务")

        print("4/5：返回垂直向下准备位。")
        _pose, ready_return_joints = move_joint_checked(
            arm,
            DOWN_READY_JOINTS,
            check_error,
            "返回准备位运动",
        )
        check_arrival(
            arm,
            ready_return_joints,
            DOWN_READY_JOINTS,
            ready_pose,
            "返回准备位",
        )

        print("5/5：返回六关节零位。")
        _pose, zero_actual_joints = move_joint_checked(
            arm,
            ZERO_JOINTS,
            check_error,
            "最终回零运动",
        )
        check_arrival(
            arm,
            zero_actual_joints,
            ZERO_JOINTS,
            zero_pose,
            "最终零位",
            require_down=False,
        )
        print("高空 yaw 对齐测试完成：夹爪保持张开，机械臂已经回零。")
    except KeyboardInterrupt:
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
# 5. 命令行入口
# =============================================================================

def parse_args():
    """读取终端参数，并返回 argparse.Namespace。"""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--x",
        type=float,
        required=True,
        help="脚本 11 输出的机械臂平面中心 X，单位米。",
    )
    parser.add_argument(
        "--y",
        type=float,
        required=True,
        help="脚本 11 输出的机械臂平面中心 Y，单位米。",
    )
    parser.add_argument(
        "--tube-yaw-deg",
        type=float,
        required=True,
        help="脚本 11 输出的机械臂基座 B→C yaw，单位度。",
    )
    parser.add_argument(
        "--yaw-offset-deg",
        type=float,
        default=90.0,
        help="试管方向到夹爪参考方向的偏移，默认 90°。",
    )
    parser.add_argument(
        "--align-z",
        type=float,
        default=DEFAULT_ALIGN_Z_M,
        help="高空对齐 TCP 高度，默认 0.300 m，限制 0.250..0.450 m。",
    )
    parser.add_argument(
        "--ip",
        default=DEFAULT_IP,
        help=f"CArm IP，默认 {DEFAULT_IP}。",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="连接 CArm 执行高空测试；省略时只打印计划。",
    )
    return parser.parse_args()


def validate_args(args) -> None:
    """在连接设备前检查命令行参数范围和组合。"""

    values = (
        args.x,
        args.y,
        args.tube_yaw_deg,
        args.yaw_offset_deg,
        args.align_z,
    )
    if not np.isfinite(values).all():
        raise ValueError("x、y、yaw 和 align-z 必须是有限数字")
    if not (-0.5 <= args.x <= 0.5 and -0.5 <= args.y <= 0.5):
        raise ValueError("x/y 超出本测试允许的 -0.5..0.5 m")
    if not MIN_ALIGN_Z_M <= args.align_z <= MAX_ALIGN_Z_M:
        raise ValueError(
            f"--align-z 必须位于 {MIN_ALIGN_Z_M:.3f}.."
            f"{MAX_ALIGN_Z_M:.3f} m"
        )


def main() -> int:
    """按照本脚本的编号流程依次执行各个步骤。"""

    args = parse_args()
    validate_args(args)
    candidates = make_candidates(
        args.x,
        args.y,
        args.align_z,
        args.tube_yaw_deg,
        args.yaw_offset_deg,
    )
    print_plan(args, candidates)
    if args.execute:
        execute_alignment(args, candidates)
    else:
        print("[DRY-RUN] 没有导入 CArm，也没有连接或发送任何命令。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("\n用户中断，高空对齐测试已经停止。") from None
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"程序终止：{exc}") from None
