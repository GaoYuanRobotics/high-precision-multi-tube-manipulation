#!/usr/bin/env python3
"""RealSense 检测黄色试管，动态对齐夹爪 yaw，抓取后抬高悬停。

流程很短：

    检测 y-body/y-cap -> 计算试管中心和 B→C 方向
    -> 转为机械臂基座 XY/yaw -> 准备位 -> 高位对齐
    -> 保持 yaw 下降 -> 等待 3 秒 -> 闭爪 -> 保持 yaw 抬高

默认只显示相机画面和打印计划。添加 ``--execute`` 并输入完整确认文字才会
连接 CArm。脚本最后停在高位并保持闭爪，不转竖直、不放置、也不自动回零。
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
# =============================================================================
# 本脚本自带的 CArm、试管几何、模型类别和 RealSense 实现
# =============================================================================

"""CArm 控制与软件安全门包装器。

官方 CArm SDK 位姿顺序为：
    [x, y, z, qx, qy, qz, qw]

位置单位为米，角度单位为弧度。工作空间、ready、单步距离和现场确认门只能
降低误操作风险，不能替代厂家碰撞检测、路径规划、现场急停或人工监护。
"""


import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple


POSITION_MODE = 1
DEFAULT_IP = "10.42.0.101"


def strict_bool(value: Any, name: str) -> bool:
    """只接受真正的 YAML/Python bool，拒绝会被误判的字符串和数字。"""

    if type(value) is bool:
        return value
    raise ValueError(
        f"{name} must be an unquoted YAML boolean true/false, "
        f"got {value!r} ({type(value).__name__})."
    )


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """配置节点必须是真正的 mapping，不能用 null/list/string 静默代替。"""

    if not isinstance(value, Mapping):
        raise ValueError(
            f"{name} must be a YAML mapping, got "
            f"{value!r} ({type(value).__name__})."
        )
    return value


def _mapping_section(data: Mapping[str, Any], key: str, name: str) -> Mapping[str, Any]:
    """读取可选 mapping 节点；节点一旦出现就必须是 mapping。"""

    if key not in data:
        return {}
    return _require_mapping(data[key], name)


def _reject_unknown_keys(
    data: Mapping[str, Any],
    allowed: set[str],
    name: str,
) -> None:
    """拒绝拼错但会被默认值掩盖的安全配置字段。"""

    unknown: list[str] = []
    for key in data:
        if key not in allowed:
            unknown.append(str(key))
    unknown.sort()
    if unknown:
        raise ValueError(f"{name} contains unknown field(s): {', '.join(unknown)}.")


def _strict_float(value: Any, name: str) -> float:
    """只接受 YAML 数字，拒绝 bool 和带引号的数字字符串。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{name} must be a YAML number, got "
            f"{value!r} ({type(value).__name__})."
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return result


def _strict_int(value: Any, name: str) -> int:
    """只接受真正的 YAML/Python int，拒绝 bool、float 和字符串。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} must be a YAML integer, got "
            f"{value!r} ({type(value).__name__})."
        )
    return value


class CarmCommandError(RuntimeError):
    """CArm SDK 拒绝命令或执行失败时抛出的统一异常。"""


@dataclass(frozen=True)
class PoseXYZQuat:
    """保存 CArm SDK 使用的笛卡尔位姿 [x,y,z,qx,qy,qz,qw]。"""

    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float

    def __post_init__(self) -> None:
        """对象创建后立即校验字段形状、单位和有限值。"""

        raw_values = (
            self.x,
            self.y,
            self.z,
            self.qx,
            self.qy,
            self.qz,
            self.qw,
        )
        float_values: list[float] = []
        for value in raw_values:
            float_values.append(float(value))
        values = tuple(float_values)
        assert_finite(values, "CArm pose")
        quaternion = normalize_quat(values[3:])
        position_names = ("x", "y", "z")
        for index in range(len(position_names)):
            object.__setattr__(self, position_names[index], values[index])
        quaternion_names = ("qx", "qy", "qz", "qw")
        for index in range(len(quaternion_names)):
            object.__setattr__(self, quaternion_names[index], quaternion[index])

    def as_list(self) -> list[float]:
        """按 SDK 约定顺序返回普通 Python 列表。"""

        return [self.x, self.y, self.z, self.qx, self.qy, self.qz, self.qw]

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "PoseXYZQuat":
        """从数值序列构造并校验当前数据对象。"""

        if len(values) != 7:
            raise ValueError("CArm pose must be [x, y, z, qx, qy, qz, qw].")
        float_values: list[float] = []
        for value in values:
            float_values.append(float(value))
        return cls(
            float_values[0],
            float_values[1],
            float_values[2],
            float_values[3],
            float_values[4],
            float_values[5],
            float_values[6],
        )


def default_accepted_response_values() -> set[str]:
    """返回当前 CArm SDK 可接受的明确成功响应。"""

    return {
        "Task_Recieve",
        "Task_Receive",
        "Task_Accept",
        "Task_Accepted",
        "OK",
        "ok",
    }


@dataclass
class CarmClientConfig:
    """保存 CArm 包装器的运行参数。

    ``down_quat_xyzw`` 表示基座坐标系中 yaw=0 时夹爪向下的标定姿态；默认值
    只是一种常见起始姿态，真实抓取前仍需在安全高处验证。
    """

    ip: str = DEFAULT_IP
    arm_index: int = 0
    connect_timeout_s: float = 3.0
    control_mode: int = POSITION_MODE
    speed_level: float = 2.0
    speed_response_level: int = 80
    tool_index: int = 0
    down_quat_xyzw: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    gripper_yaw_offset_rad: float = 0.0
    gripper_open_m: float = 0.060
    gripper_close_m: float = 0.018
    gripper_tau_n: float = 10.0
    z_safe_m: float = 0.300
    z_grasp_m: float = 0.080
    z_insert_m: float = 0.065
    settle_s: float = 0.2
    setup_verified: bool = False
    workspace_x_min_m: float = -0.650
    workspace_x_max_m: float = 0.650
    workspace_y_min_m: float = -0.650
    workspace_y_max_m: float = 0.650
    workspace_z_min_m: float = 0.050
    workspace_z_max_m: float = 0.650
    max_single_step_m: float = 0.250
    max_single_rotation_deg: float = 30.0
    motion_timeout_s: float = 30.0
    motion_stop_grace_s: float = 2.0
    motion_completion_position_tolerance_m: float = 0.005
    motion_completion_rotation_tolerance_deg: float = 3.0
    require_ready_for_actions: bool = True
    collision_configure_on_ready: bool = False
    collision_enabled: bool = True
    collision_sensitivity_level: int = 1
    collision_required_for_motion: bool = False
    allow_unverified_vertical_place: bool = False
    require_accepted_response: bool = True
    accepted_response_values: set[str] = field(
        default_factory=default_accepted_response_values
    )

    def __post_init__(self) -> None:
        """尽早拒绝无效安全配置，避免连接硬件后才暴露问题。"""

        for name in (
            "setup_verified",
            "require_ready_for_actions",
            "collision_configure_on_ready",
            "collision_enabled",
            "collision_required_for_motion",
            "allow_unverified_vertical_place",
            "require_accepted_response",
        ):
            strict_bool(getattr(self, name), name)
        if not isinstance(self.ip, str) or not self.ip.strip():
            raise ValueError("CArm ip must be a non-empty string.")
        for name in ("arm_index", "tool_index", "speed_response_level"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        self.down_quat_xyzw = normalize_quat(self.down_quat_xyzw)

        bounds = (
            ("x", self.workspace_x_min_m, self.workspace_x_max_m),
            ("y", self.workspace_y_min_m, self.workspace_y_max_m),
            ("z", self.workspace_z_min_m, self.workspace_z_max_m),
        )
        for axis, lower, upper in bounds:
            assert_finite((lower, upper), f"workspace {axis} bounds")
            if lower >= upper:
                raise ValueError(
                    f"workspace {axis} lower bound must be smaller than upper bound."
                )

        assert_finite(
            (
                self.max_single_step_m,
                self.max_single_rotation_deg,
                self.motion_timeout_s,
                self.motion_stop_grace_s,
                self.motion_completion_position_tolerance_m,
                self.motion_completion_rotation_tolerance_deg,
                self.gripper_open_m,
                self.gripper_close_m,
                self.gripper_tau_n,
                self.settle_s,
                self.connect_timeout_s,
                self.speed_level,
                self.gripper_yaw_offset_rad,
                self.z_safe_m,
                self.z_grasp_m,
                self.z_insert_m,
            ),
            "CArm safety config",
        )
        if self.max_single_step_m <= 0:
            raise ValueError("max_single_step_m must be greater than zero.")
        if not 0.0 < self.max_single_rotation_deg <= 180.0:
            raise ValueError(
                "max_single_rotation_deg must be within (0, 180]."
            )
        if self.motion_timeout_s <= 0.0:
            raise ValueError("motion timeout_s must be greater than zero.")
        if self.motion_stop_grace_s < 0.0:
            raise ValueError("motion stop_grace_s must not be negative.")
        if not 0.0 < self.motion_completion_position_tolerance_m <= 0.050:
            raise ValueError(
                "motion completion_position_tolerance_m must be within "
                "(0.000, 0.050] m."
            )
        if not 0.0 < self.motion_completion_rotation_tolerance_deg <= 15.0:
            raise ValueError(
                "motion completion_rotation_tolerance_deg must be within "
                "(0.0, 15.0] deg."
            )
        if self.connect_timeout_s <= 0.0:
            raise ValueError("connect_timeout_s must be greater than zero.")
        if self.speed_level <= 0.0:
            raise ValueError("speed_level must be greater than zero.")
        if not 0.0 <= self.gripper_open_m <= 0.080:
            raise ValueError("gripper open_m must be within [0.000, 0.080] m.")
        if not 0.0 <= self.gripper_close_m <= 0.080:
            raise ValueError("gripper close_m must be within [0.000, 0.080] m.")
        if self.gripper_open_m <= self.gripper_close_m:
            raise ValueError("gripper open_m must be greater than close_m.")
        if not 0.0 <= self.gripper_tau_n <= 100.0:
            raise ValueError("gripper tau_n must be within [0, 100] N.")
        if self.settle_s < 0:
            raise ValueError("motion settle_s must not be negative.")
        for name, value in (
            ("z_safe_m", self.z_safe_m),
            ("z_grasp_m", self.z_grasp_m),
            ("z_insert_m", self.z_insert_m),
        ):
            if not self.workspace_z_min_m <= value <= self.workspace_z_max_m:
                raise ValueError(
                    f"{name} must be inside the configured Z workspace."
                )
        if self.z_safe_m <= max(self.z_grasp_m, self.z_insert_m):
            raise ValueError(
                "z_safe_m must be greater than z_grasp_m and z_insert_m."
            )
        if not 0 <= self.collision_sensitivity_level <= 2:
            raise ValueError("collision sensitivity_level must be 0, 1, or 2.")
        if self.collision_required_for_motion and not self.collision_configure_on_ready:
            raise ValueError(
                "collision.required_for_motion=true requires "
                "collision.configure_on_ready=true."
            )
        if not self.require_ready_for_actions:
            raise ValueError(
                "require_ready_for_actions is a mandatory safety gate and "
                "cannot be disabled."
            )
        if not self.require_accepted_response:
            raise ValueError(
                "require_accepted_response is a mandatory safety gate for the "
                "pinned CArm SDK and cannot be disabled."
            )
        response_values_are_valid = True
        for value in self.accepted_response_values:
            if not isinstance(value, str) or not value:
                response_values_are_valid = False
                break
        if not self.accepted_response_values or not response_values_are_valid:
            raise ValueError(
                "accepted_response_values must contain non-empty strings."
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CarmClientConfig":
        """从字典配置构造并严格校验当前对象。"""

        data = _require_mapping(data, "CArm config")
        _reject_unknown_keys(
            data,
            {
                "ip",
                "arm_index",
                "connect_timeout_s",
                "control_mode",
                "speed_level",
                "speed_response_level",
                "tool_index",
                "down_quat_xyzw",
                "gripper_yaw_offset_rad",
                "gripper_yaw_offset_deg",
                "gripper",
                "motion",
                "safety",
                "require_accepted_response",
            },
            "CArm config",
        )
        gripper = _mapping_section(data, "gripper", "gripper")
        motion = _mapping_section(data, "motion", "motion")
        safety = _mapping_section(data, "safety", "safety")
        workspace = _mapping_section(safety, "workspace", "safety.workspace")
        collision = _mapping_section(safety, "collision", "safety.collision")
        _reject_unknown_keys(gripper, {"open_m", "close_m", "tau_n"}, "gripper")
        _reject_unknown_keys(
            motion,
            {
                "z_safe_m",
                "z_grasp_m",
                "z_insert_m",
                "settle_s",
                "timeout_s",
                "stop_grace_s",
                "completion_position_tolerance_m",
                "completion_rotation_tolerance_deg",
            },
            "motion",
        )
        _reject_unknown_keys(
            safety,
            {
                "setup_verified",
                "workspace",
                "max_single_step_m",
                "max_single_rotation_deg",
                "require_ready_for_actions",
                "collision",
                "allow_unverified_vertical_place",
            },
            "safety",
        )
        _reject_unknown_keys(
            workspace,
            {
                "x_min_m",
                "x_max_m",
                "y_min_m",
                "y_max_m",
                "z_min_m",
                "z_max_m",
            },
            "safety.workspace",
        )
        _reject_unknown_keys(
            collision,
            {
                "configure_on_ready",
                "enabled",
                "sensitivity_level",
                "required_for_motion",
            },
            "safety.collision",
        )

        if (
            "gripper_yaw_offset_rad" in data
            and "gripper_yaw_offset_deg" in data
        ):
            raise ValueError(
                "CArm config cannot contain both gripper_yaw_offset_rad and "
                "gripper_yaw_offset_deg."
            )
        if "gripper_yaw_offset_rad" in data:
            yaw_offset_rad = _strict_float(
                data["gripper_yaw_offset_rad"],
                "gripper_yaw_offset_rad",
            )
        else:
            yaw_offset_rad = math.radians(
                _strict_float(
                    data.get("gripper_yaw_offset_deg", 0.0),
                    "gripper_yaw_offset_deg",
                )
            )

        down_quat = data.get("down_quat_xyzw", (1.0, 0.0, 0.0, 0.0))
        if (
            not isinstance(down_quat, Sequence)
            or isinstance(down_quat, (str, bytes))
            or len(down_quat) != 4
        ):
            raise ValueError(
                "down_quat_xyzw must be a four-element YAML sequence."
            )
        down_quat_list: list[float] = []
        for index in range(len(down_quat)):
            value = down_quat[index]
            down_quat_list.append(
                _strict_float(value, f"down_quat_xyzw[{index}]")
            )
        down_quat_values = tuple(down_quat_list)
        ip = data.get("ip", DEFAULT_IP)
        if not isinstance(ip, str):
            raise ValueError("ip must be a YAML string.")

        return cls(
            ip=ip,
            arm_index=_strict_int(data.get("arm_index", 0), "arm_index"),
            connect_timeout_s=_strict_float(
                data.get("connect_timeout_s", 3.0),
                "connect_timeout_s",
            ),
            control_mode=_strict_int(
                data.get("control_mode", POSITION_MODE),
                "control_mode",
            ),
            speed_level=_strict_float(
                data.get("speed_level", 2.0),
                "speed_level",
            ),
            speed_response_level=_strict_int(
                data.get("speed_response_level", 80),
                "speed_response_level",
            ),
            tool_index=_strict_int(data.get("tool_index", 0), "tool_index"),
            down_quat_xyzw=normalize_quat(down_quat_values),
            gripper_yaw_offset_rad=yaw_offset_rad,
            gripper_open_m=_strict_float(
                gripper.get("open_m", 0.060),
                "gripper.open_m",
            ),
            gripper_close_m=_strict_float(
                gripper.get("close_m", 0.018),
                "gripper.close_m",
            ),
            gripper_tau_n=_strict_float(
                gripper.get("tau_n", 10.0),
                "gripper.tau_n",
            ),
            z_safe_m=_strict_float(
                motion.get("z_safe_m", 0.300),
                "motion.z_safe_m",
            ),
            z_grasp_m=_strict_float(
                motion.get("z_grasp_m", 0.080),
                "motion.z_grasp_m",
            ),
            z_insert_m=_strict_float(
                motion.get("z_insert_m", 0.065),
                "motion.z_insert_m",
            ),
            settle_s=_strict_float(
                motion.get("settle_s", 0.2),
                "motion.settle_s",
            ),
            setup_verified=strict_bool(
                safety.get("setup_verified", False),
                "safety.setup_verified",
            ),
            workspace_x_min_m=_strict_float(
                workspace.get("x_min_m", -0.650),
                "safety.workspace.x_min_m",
            ),
            workspace_x_max_m=_strict_float(
                workspace.get("x_max_m", 0.650),
                "safety.workspace.x_max_m",
            ),
            workspace_y_min_m=_strict_float(
                workspace.get("y_min_m", -0.650),
                "safety.workspace.y_min_m",
            ),
            workspace_y_max_m=_strict_float(
                workspace.get("y_max_m", 0.650),
                "safety.workspace.y_max_m",
            ),
            workspace_z_min_m=_strict_float(
                workspace.get("z_min_m", 0.050),
                "safety.workspace.z_min_m",
            ),
            workspace_z_max_m=_strict_float(
                workspace.get("z_max_m", 0.650),
                "safety.workspace.z_max_m",
            ),
            max_single_step_m=_strict_float(
                safety.get("max_single_step_m", 0.250),
                "safety.max_single_step_m",
            ),
            max_single_rotation_deg=_strict_float(
                safety.get("max_single_rotation_deg", 30.0),
                "safety.max_single_rotation_deg",
            ),
            motion_timeout_s=_strict_float(
                motion.get("timeout_s", 30.0),
                "motion.timeout_s",
            ),
            motion_stop_grace_s=_strict_float(
                motion.get("stop_grace_s", 2.0),
                "motion.stop_grace_s",
            ),
            motion_completion_position_tolerance_m=_strict_float(
                motion.get("completion_position_tolerance_m", 0.005),
                "motion.completion_position_tolerance_m",
            ),
            motion_completion_rotation_tolerance_deg=_strict_float(
                motion.get("completion_rotation_tolerance_deg", 3.0),
                "motion.completion_rotation_tolerance_deg",
            ),
            require_ready_for_actions=strict_bool(
                safety.get("require_ready_for_actions", True),
                "safety.require_ready_for_actions",
            ),
            collision_configure_on_ready=strict_bool(
                collision.get("configure_on_ready", False),
                "safety.collision.configure_on_ready",
            ),
            collision_enabled=strict_bool(
                collision.get("enabled", True),
                "safety.collision.enabled",
            ),
            collision_sensitivity_level=_strict_int(
                collision.get("sensitivity_level", 1),
                "safety.collision.sensitivity_level",
            ),
            collision_required_for_motion=strict_bool(
                collision.get("required_for_motion", False),
                "safety.collision.required_for_motion",
            ),
            allow_unverified_vertical_place=strict_bool(
                safety.get("allow_unverified_vertical_place", False),
                "safety.allow_unverified_vertical_place",
            ),
            require_accepted_response=strict_bool(
                data.get("require_accepted_response", True),
                "require_accepted_response",
            ),
        )


def assert_finite(values: Iterable[float], name: str = "values") -> None:
    """确认输入数值全部有限，拒绝 NaN 和无穷大。"""

    bad: list[float] = []
    for value in values:
        if not math.isfinite(float(value)):
            bad.append(value)
    if bad:
        raise ValueError(f"{name} contains non-finite values: {bad}")


def wrap_pi(angle_rad: float) -> float:
    """把弧度角规范到 [-π, π) 区间。"""

    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def normalize_quat(quat_xyzw: Sequence[float]) -> Tuple[float, float, float, float]:
    """把四元数归一化，并拒绝退化输入。"""

    if len(quat_xyzw) != 4:
        raise ValueError("Quaternion must be [qx, qy, qz, qw].")
    quaternion_values: list[float] = []
    for value in quat_xyzw:
        quaternion_values.append(float(value))
    qx, qy, qz, qw = quaternion_values
    assert_finite((qx, qy, qz, qw), "quaternion")
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        raise ValueError("Quaternion norm is zero.")
    return (qx / norm, qy / norm, qz / norm, qw / norm)


def quat_angular_distance_rad(
    first_xyzw: Sequence[float],
    second_xyzw: Sequence[float],
) -> float:
    """返回两个姿态的最小旋转角，正确处理 q 与 -q 的等价性。"""

    first = normalize_quat(first_xyzw)
    second = normalize_quat(second_xyzw)
    dot = 0.0
    for index in range(4):
        dot += first[index] * second[index]
    dot = abs(dot)
    return 2.0 * math.acos(min(max(dot, 0.0), 1.0))


def quat_multiply(
    left_xyzw: Sequence[float],
    right_xyzw: Sequence[float],
) -> Tuple[float, float, float, float]:
    """按照 [qx,qy,qz,qw] 顺序计算四元数 Hamilton 乘积。"""

    ax, ay, az, aw = normalize_quat(left_xyzw)
    bx, by, bz, bw = normalize_quat(right_xyzw)
    return normalize_quat(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    )


def yaw_quat(yaw_rad: float) -> Tuple[float, float, float, float]:
    """生成绕基座 Z 轴旋转指定 yaw 的四元数。"""

    half = 0.5 * float(yaw_rad)
    return (0.0, 0.0, math.sin(half), math.cos(half))


def down_pose_quat(
    yaw_rad: float,
    down_quat_xyzw: Sequence[float],
    yaw_offset_rad: float = 0.0,
) -> Tuple[float, float, float, float]:
    """生成夹爪向下并绕基座 Z 轴带指定 yaw 的四元数。"""

    yaw = wrap_pi(float(yaw_rad) + float(yaw_offset_rad))
    return quat_multiply(yaw_quat(yaw), down_quat_xyzw)


class CarmClient:
    """项目级 CArm 控制接口。

    这里统一处理连接、SDK 响应、夹爪单位、XYZ+yaw 到四元数的转换，以及
    工作空间/ready/单步距离门禁。当前 01–10 主流程不直接使用本包装器；
    ``place_at()`` 默认封锁，因为它不包含水平试管转竖直插管所需的完整
    姿态路径。
    """

    def __init__(self, config: Optional[CarmClientConfig] = None):
        """初始化当前对象，并保存后续操作需要的状态。"""

        self.config = config or CarmClientConfig()
        self._arm: Any = None
        # 只在本次连接中 ready() 全部成功后置 True；断开、停止或异常后清零。
        self._ready_confirmed = False
        self._collision_confirmed = False
        self._action_started = False

    def __enter__(self) -> "CarmClient":
        """进入 with 代码块时返回当前设备客户端。"""

        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        # 只有已经开始过状态改变/运动动作且发生异常时才发停止命令。
        # 纯只读连接异常不主动改变控制器状态。
        """离开 with 代码块时释放设备连接。"""

        if exc_type is not None and self._action_started:
            self.stop_best_effort()
        self.disconnect()

    @property
    def arm(self) -> Any:
        """返回已经创建的底层 CArm SDK 对象。"""

        if self._arm is None:
            raise CarmCommandError("CArm is not connected.")
        return self._arm

    def connect(self) -> bool:
        """连接机械臂控制器，并等待 WebSocket 状态就绪。"""

        if self.is_connected():
            return True

        try:
            from carm import Carm
        except ImportError as exc:
            raise CarmCommandError(
                "Python package 'carm' is not installed. Run `pip install carm` "
                "inside the project environment."
            ) from exc

        self._arm = Carm(addr=self.config.ip, arm_index=self.config.arm_index)

        deadline = time.monotonic() + self.config.connect_timeout_s
        while time.monotonic() < deadline:
            if self.is_connected():
                return True
            time.sleep(0.05)

        ok = bool(self._arm.connect(timeout=self.config.connect_timeout_s))
        if not ok:
            self.disconnect()
            raise CarmCommandError(f"Failed to connect to CArm at {self.config.ip}.")
        return True

    def disconnect(self) -> None:
        """断开 CArm WebSocket 连接。"""

        if self._arm is None:
            return
        try:
            self._arm.disconnect()
        finally:
            self._arm = None
            self._ready_confirmed = False
            self._collision_confirmed = False
            self._action_started = False

    def is_connected(self) -> bool:
        """返回 CArm SDK 当前是否已经连接。"""

        if self._arm is None:
            return False
        try:
            return bool(self._arm.is_connected())
        except Exception:
            return False

    def ready(self) -> None:
        """清错、使能并配置速度/工具；成功后才打开本地动作门禁。"""

        self._ready_confirmed = False
        self._collision_confirmed = False
        try:
            def send_ready_command() -> Any:
                return self.arm.set_ready()

            self._run_action("set_ready", send_ready_command)
            self.set_control_mode(self.config.control_mode)
            self.set_speed_level(
                self.config.speed_level,
                self.config.speed_response_level,
            )
            self.set_tool_index(self.config.tool_index)
            reported_tool = self.current_tool_index()
            if reported_tool is None:
                raise CarmCommandError(
                    "CArm state does not explicitly report the active tool after "
                    "set_tool_index(); motion cannot be enabled safely."
                )
            if reported_tool != self.config.tool_index:
                raise CarmCommandError(
                    "CArm reported tool_index="
                    f"{reported_tool}, expected {self.config.tool_index} after ready()."
                )

            # 本机旧 SDK 有该接口，但项目尚未完成跨版本现场验证，因此默认关闭。
            # 一旦配置要求调用，接口缺失或返回失败都会阻止 ready 成功。
            if self.config.collision_configure_on_ready:
                method = getattr(self.arm, "set_collision_config", None)
                if not callable(method):
                    raise CarmCommandError(
                        "Collision gate was requested, but this CArm SDK does not "
                        "provide set_collision_config()."
                    )
                def send_collision_command() -> Any:
                    return method(
                        bool(self.config.collision_enabled),
                        int(self.config.collision_sensitivity_level),
                    )

                self._run_action("set_collision_config", send_collision_command)
                self._collision_confirmed = True

            sdk_state = self.sdk_ready_state()
            if sdk_state is False:
                raise CarmCommandError(
                    "CArm SDK state does not report servo=1, POSITION mode, "
                    "and a non-error controller state after ready()."
                )
            # 某些 SDK 版本不公开完整状态字典；此时只能依据每条命令的成功响应。
            self._ready_confirmed = True
        except Exception:
            self.stop_best_effort()
            raise

    @property
    def ready_confirmed(self) -> bool:
        """本次连接是否已经由本包装器成功执行过 ready()。"""

        return bool(self._ready_confirmed and self.is_connected())

    @property
    def collision_confirmed(self) -> bool:
        """本次连接是否成功调用过 SDK 碰撞检测配置。"""

        return bool(self._collision_confirmed and self.is_connected())

    def sdk_ready_state(self) -> Optional[bool]:
        """尽力读取 SDK 状态；接口不存在时返回 None，不伪造“已就绪”。"""

        state = getattr(self.arm, "_arm_state", None)
        if not isinstance(state, Mapping) or not state:
            return None
        return bool(
            state.get("servo", 0) == 1
            and state.get("fsm_state", "") == "POSITION"
            and state.get("state", -1) != -1
        )

    def set_control_mode(self, mode: int = POSITION_MODE) -> None:
        """调用 CArm SDK 的 set_control_mode 命令，并返回原始响应。"""

        def send_control_mode_command() -> Any:
            return self.arm.set_control_mode(int(mode))

        self._run_action("set_control_mode", send_control_mode_command)

    def set_speed_level(self, level: float, response_level: int = 80) -> None:
        """调用 CArm SDK 的 set_speed_level 命令，并返回原始响应。"""

        def send_speed_command() -> Any:
            return self.arm.set_speed_level(float(level), int(response_level))

        self._run_action("set_speed_level", send_speed_command)

    def set_tool_index(self, index: int) -> None:
        """调用 CArm SDK 的 set_tool_index 命令，并返回原始响应。"""

        def send_tool_command() -> Any:
            return self.arm.set_tool_index(int(index))

        self._run_action("set_tool_index", send_tool_command)

    def current_pose(self) -> PoseXYZQuat:
        """读取 SDK 上报的实际法兰位姿，不把它误称为工具 TCP 位姿。"""

        pose = self.arm.cart_pose
        try:
            return PoseXYZQuat.from_sequence(pose)
        except (TypeError, ValueError) as exc:
            raise CarmCommandError(
                f"CArm returned an invalid current cart_pose: {pose!r}"
            ) from exc

    def current_tool_pose(self) -> PoseXYZQuat:
        """由实际关节角和当前配置工具号正解得到 TCP 位姿。

        CArm ``cart_pose`` 明确表示法兰，而 ``move_pose(..., tool=N)`` 的
        目标按工具 N 的 TCP 解释。二者不能直接计算单步距离。本方法强制使用
        SDK ``forward_kine(joint_pos, tool=N)`` 取得同一 TCP 参考点；接口缺失、
        当前关节无效或正解失败时，真实运动会被拒绝。
        """

        try:
            joints = list(self.arm.joint_pos)
        except (AttributeError, TypeError, ValueError) as exc:
            raise CarmCommandError(
                "Cannot read current joint_pos for configured-TCP forward kinematics."
            ) from exc
        if not joints:
            raise CarmCommandError(
                "Current joint_pos is empty; configured-TCP step check cannot run."
            )
        try:
            assert_finite(joints, "current joint_pos")
        except ValueError as exc:
            raise CarmCommandError(str(exc)) from exc
        method = getattr(self.arm, "forward_kine", None)
        if not callable(method):
            raise CarmCommandError(
                "This CArm SDK does not provide forward_kine(); motion is blocked "
                "because flange cart_pose cannot be compared with a tool-TCP target."
            )
        pose: Any = None
        try:
            pose = method(joints, tool=self.config.tool_index)
            return PoseXYZQuat.from_sequence(pose)
        except Exception as exc:
            raise CarmCommandError(
                "CArm forward_kine() did not return a valid configured-TCP pose: "
                f"{pose!r}."
            ) from exc

    def current_tool_index(self) -> Optional[int]:
        """读取控制器明确上报的工具号；缺少原始状态证据时返回 None。"""

        state = getattr(self.arm, "_arm_state", None)
        if not isinstance(state, Mapping) or "tool" not in state:
            return None
        try:
            value = getattr(self.arm, "tool_index")
        except (AttributeError, TypeError, ValueError):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def tool_coordinate(self, index: Optional[int] = None) -> Any:
        """读取工具相对法兰的坐标；SDK 无该接口时返回 None。"""

        method = getattr(self.arm, "get_tool_coordinate", None)
        if not callable(method):
            return None
        tool_index = self.config.tool_index
        if index is not None:
            tool_index = int(index)
        return method(tool_index)

    def move_pose(
        self,
        pose: PoseXYZQuat | Sequence[float],
        *,
        is_sync: bool = True,
        tool: Optional[int] = None,
    ) -> None:
        """完成 CarmClient.move_pose 对应的操作。"""

        self._require_motion_ready()
        if not is_sync:
            raise CarmCommandError(
                "Asynchronous motion is blocked because this wrapper cannot "
                "verify completion before a following command."
            )
        tool_index = self.config.tool_index
        if tool is not None:
            tool_index = int(tool)
        if tool_index != self.config.tool_index:
            raise CarmCommandError(
                f"Motion tool={tool_index} does not match configured/verified "
                f"tool_index={self.config.tool_index}."
            )
        pose_list = self._pose_list(pose)
        self.validate_pose_target(pose_list)
        def send_move_pose_command() -> Any:
            return self.arm.move_pose(
                pose_list,
                is_sync=True,
                tool=tool_index,
            )

        self._run_motion_action(
            "move_pose",
            send_move_pose_command,
            target_pose=pose_list,
        )

    def move_line_pose(
        self,
        pose: PoseXYZQuat | Sequence[float],
        *,
        is_sync: bool = True,
        tool: Optional[int] = None,
    ) -> None:
        """完成 CarmClient.move_line_pose 对应的操作。"""

        self._require_motion_ready()
        if not is_sync:
            raise CarmCommandError(
                "Asynchronous motion is blocked because this wrapper cannot "
                "verify completion before a following command."
            )
        tool_index = self.config.tool_index
        if tool is not None:
            tool_index = int(tool)
        if tool_index != self.config.tool_index:
            raise CarmCommandError(
                f"Motion tool={tool_index} does not match configured/verified "
                f"tool_index={self.config.tool_index}."
            )
        pose_list = self._pose_list(pose)
        self.validate_pose_target(pose_list)
        def send_move_line_command() -> Any:
            return self.arm.move_line_pose(
                pose_list,
                is_sync=True,
                tool=tool_index,
            )

        self._run_motion_action(
            "move_line_pose",
            send_move_line_command,
            target_pose=pose_list,
        )

    def move_joints(self, joints_rad: Sequence[float]) -> None:
        """同步移动到六关节目标，并用 tool TCP 的 FK 结果验收到位。

        这个小接口主要用于从已确认的关节零位进入固定准备位。普通视觉目标仍应
        使用 ``move_pose`` / ``move_line_pose``，不要把视觉位置硬编码成关节角。
        """

        self._require_motion_ready()
        joints: list[float] = []
        for value in joints_rad:
            joints.append(float(value))
        if len(joints) != 6:
            raise ValueError("CArm joint target must contain exactly 6 values.")
        assert_finite(joints, "joint target")

        method = getattr(self.arm, "forward_kine", None)
        if not callable(method):
            raise CarmCommandError(
                "This CArm SDK does not provide forward_kine(); joint motion "
                "cannot be verified at the configured tool TCP."
            )
        target_pose = PoseXYZQuat.from_sequence(
            method(joints, tool=self.config.tool_index)
        )
        self.validate_pose_target(target_pose)
        def send_move_joint_command() -> Any:
            return self.arm.move_joint(joints, is_sync=True)

        self._run_motion_action(
            "move_joint",
            send_move_joint_command,
            target_pose=target_pose,
        )

    def make_xyzyaw_pose(self, x: float, y: float, z: float, yaw_rad: float) -> PoseXYZQuat:
        """完成 CarmClient.make_xyzyaw_pose 对应的操作。"""

        assert_finite((x, y, z, yaw_rad), "xyzyaw pose")

        # 视觉给的是平面 yaw；机械臂 SDK 要的是四元数姿态。
        qx, qy, qz, qw = down_pose_quat(
            yaw_rad,
            self.config.down_quat_xyzw,
            self.config.gripper_yaw_offset_rad,
        )
        return PoseXYZQuat(float(x), float(y), float(z), qx, qy, qz, qw)

    def validate_pose_target(
        self,
        pose: PoseXYZQuat | Sequence[float],
        *,
        current_pose: Optional[PoseXYZQuat | Sequence[float]] = None,
        check_step: bool = True,
    ) -> Optional[float]:
        """检查工作空间、TCP 平移步长和姿态旋转步长；不发送运动命令。

        返回当前点到目标点的平移距离。离线预览可传 ``check_step=False``，
        此时仅检查工作空间并返回 None；真实运动不会关闭单步检查。若未显式
        提供 ``current_pose``，当前 TCP 必须通过关节角和指定工具正运动学取得，
        绝不会把 SDK 的法兰 ``cart_pose`` 与 TCP 目标混算。
        """

        target = PoseXYZQuat.from_sequence(self._pose_list(pose))
        self._validate_workspace_xyz(target.x, target.y, target.z)
        if not check_step:
            return None

        if current_pose is None:
            current = self.current_tool_pose()
        elif isinstance(current_pose, PoseXYZQuat):
            current = current_pose
        else:
            current = PoseXYZQuat.from_sequence(current_pose)

        distance_m = math.dist(
            (current.x, current.y, current.z),
            (target.x, target.y, target.z),
        )
        if distance_m > self.config.max_single_step_m + 1e-12:
            raise CarmCommandError(
                "Target translation step is too large: "
                f"{distance_m:.4f} m > max_single_step_m="
                f"{self.config.max_single_step_m:.4f} m."
            )
        rotation_rad = quat_angular_distance_rad(
            (current.qx, current.qy, current.qz, current.qw),
            (target.qx, target.qy, target.qz, target.qw),
        )
        rotation_deg = math.degrees(rotation_rad)
        if rotation_deg > self.config.max_single_rotation_deg + 1e-9:
            raise CarmCommandError(
                "Target orientation step is too large: "
                f"{rotation_deg:.2f} deg > max_single_rotation_deg="
                f"{self.config.max_single_rotation_deg:.2f} deg."
            )
        return distance_m

    def validate_motion_plan(
        self,
        poses: Sequence[PoseXYZQuat | Sequence[float]],
        *,
        current_pose: Optional[PoseXYZQuat | Sequence[float]] = None,
    ) -> list[float]:
        """在第一个真实动作前校验整段离散路径的边界和相邻点距离。"""

        if current_pose is None:
            previous = self.current_tool_pose()
        elif isinstance(current_pose, PoseXYZQuat):
            previous = current_pose
        else:
            previous = PoseXYZQuat.from_sequence(current_pose)

        distances: list[float] = []
        for pose in poses:
            target = PoseXYZQuat.from_sequence(self._pose_list(pose))
            distance = self.validate_pose_target(
                target,
                current_pose=previous,
                check_step=True,
            )
            distances.append(float(distance))
            previous = target
        return distances

    def move_xyzyaw(
        self,
        x: float,
        y: float,
        z: float,
        yaw_rad: float,
        *,
        linear: bool = False,
        is_sync: bool = True,
    ) -> None:
        """完成 CarmClient.move_xyzyaw 对应的操作。"""

        pose = self.make_xyzyaw_pose(x, y, z, yaw_rad)
        if linear:
            self.move_line_pose(pose, is_sync=is_sync)
        else:
            self.move_pose(pose, is_sync=is_sync)

    def move_above(
        self,
        x: float,
        y: float,
        yaw_rad: float,
        *,
        z: Optional[float] = None,
        is_sync: bool = True,
    ) -> None:
        """完成 CarmClient.move_above 对应的操作。"""

        selected_z = self.config.z_safe_m
        if z is not None:
            selected_z = float(z)
        self.move_xyzyaw(
            x,
            y,
            selected_z,
            yaw_rad,
            linear=False,
            is_sync=is_sync,
        )

    def open_gripper(self) -> None:
        """完成 CarmClient.open_gripper 对应的操作。"""

        self.set_gripper(self.config.gripper_open_m, self.config.gripper_tau_n)

    def close_gripper(self) -> None:
        """完成 CarmClient.close_gripper 对应的操作。"""

        self.set_gripper(self.config.gripper_close_m, self.config.gripper_tau_n)

    def set_gripper(self, pos_m: float, tau_n: Optional[float] = None) -> None:
        """调用 CArm SDK 的 set_gripper 命令，并返回原始响应。"""

        self._require_action_ready("gripper")
        pos_m = float(pos_m)
        if tau_n is None:
            tau_n = self.config.gripper_tau_n
        else:
            tau_n = float(tau_n)
        assert_finite((pos_m, tau_n), "gripper command")
        if not 0.0 <= pos_m <= 0.080:
            raise CarmCommandError("Gripper position must be within [0.000, 0.080] m.")
        if not 0.0 <= tau_n <= 100.0:
            raise CarmCommandError("Gripper force must be within [0, 100] N.")
        def send_gripper_command() -> Any:
            return self.arm.set_gripper(pos_m, tau_n)

        self._run_action("set_gripper", send_gripper_command)

    def pick_at(
        self,
        x: float,
        y: float,
        yaw_rad: float,
        *,
        z_above: Optional[float] = None,
        z_grasp: Optional[float] = None,
    ) -> None:
        """在已验证的向下姿态下抓取；下降前只在安全高位打开夹爪。"""

        if z_above is None:
            z_above = self.config.z_safe_m
        else:
            z_above = float(z_above)
        if z_grasp is None:
            z_grasp = self.config.z_grasp_m
        else:
            z_grasp = float(z_grasp)
        if z_above <= z_grasp:
            raise CarmCommandError(
                f"pick_at requires z_above > z_grasp, got {z_above:.4f} <= "
                f"{z_grasp:.4f} m."
            )
        above = self.make_xyzyaw_pose(x, y, z_above, yaw_rad)
        grasp = self.make_xyzyaw_pose(x, y, z_grasp, yaw_rad)

        self._require_motion_ready()
        # 必须在第一条运动/夹爪命令前校验整段离散路径，避免动作执行到一半
        # 才发现抓取高度越界或相邻点过远。
        self.validate_motion_plan((above, grasp, above))

        try:
            self.move_pose(above)
            # 先到已验证的安全高位，再打开夹爪；禁止在当前位置盲目张开。
            self.open_gripper()
            time.sleep(self.config.settle_s)
            self.move_line_pose(grasp)
            time.sleep(self.config.settle_s)
            self.close_gripper()
            time.sleep(self.config.settle_s)
            self.move_line_pose(above)
        except Exception:
            self.stop_best_effort()
            raise

    def place_at(
        self,
        x: float,
        y: float,
        yaw_rad: float,
        *,
        z_above: Optional[float] = None,
        z_insert: Optional[float] = None,
    ) -> None:
        """仅用于已经竖直夹持物体的下降放置，不负责水平转竖直。

        默认配置会禁用该接口。当前项目尚未验证水平试管转为竖直插管的完整
        姿态与避障路径，因此不能把这个简化动作当成最终插管程序。
        """

        if not self.config.allow_unverified_vertical_place:
            raise CarmCommandError(
                "place_at() is blocked: it only keeps one down+yaw orientation "
                "and does not implement the required horizontal-to-vertical tube "
                "rotation. Validate a complete pose path first, then explicitly "
                "enable safety.allow_unverified_vertical_place only for an "
                "already-vertical object."
            )

        if z_above is None:
            z_above = self.config.z_safe_m
        else:
            z_above = float(z_above)
        if z_insert is None:
            z_insert = self.config.z_insert_m
        else:
            z_insert = float(z_insert)
        if z_above <= z_insert:
            raise CarmCommandError(
                f"place_at requires z_above > z_insert, got {z_above:.4f} <= "
                f"{z_insert:.4f} m."
            )
        above = self.make_xyzyaw_pose(x, y, z_above, yaw_rad)
        insert = self.make_xyzyaw_pose(x, y, z_insert, yaw_rad)

        self._require_motion_ready()
        self.validate_motion_plan((above, insert, above))
        try:
            self.move_pose(above)
            self.move_line_pose(insert)
            time.sleep(self.config.settle_s)
            self.open_gripper()
            time.sleep(self.config.settle_s)
            self.move_line_pose(above)
        except Exception:
            self.stop_best_effort()
            raise

    def stop_best_effort(self) -> bool:
        """异常后的尽力停止；不掩盖原始异常，也不把失败误报成成功。"""

        if self._arm is None or not self.is_connected():
            self._ready_confirmed = False
            self._collision_confirmed = False
            return False

        called = False
        succeeded = False
        stop_task = getattr(self._arm, "stop_task", None)
        if callable(stop_task):
            called = True
            try:
                stop_task(at_once=True)
                succeeded = True
            except Exception:
                pass

        stop = getattr(self._arm, "stop", None)
        if callable(stop):
            called = True
            try:
                # type=1 是 SDK 文档中的“停止”；不自动发禁用或急停命令。
                stop(1)
                succeeded = True
            except Exception:
                pass

        self._ready_confirmed = False
        self._collision_confirmed = False
        self._action_started = False
        return bool(called and succeeded)

    def _require_action_ready(self, action_name: str) -> None:
        """完成 _require_action_ready 对应的内部校验或数据转换。"""

        if not self.config.setup_verified:
            raise CarmCommandError(
                f"{action_name} is blocked because safety.setup_verified=false. "
                "Verify tool/TCP, down quaternion, yaw offset, Z heights, and "
                "workspace bounds on site before changing it to true."
            )
        if self.config.require_ready_for_actions and not self.ready_confirmed:
            raise CarmCommandError(
                f"{action_name} is blocked: call ready() successfully in this "
                "connection before any real action."
            )
        # 本地 gate 只证明本次连接曾成功 ready；每条 actuator 命令前仍要重新
        # 查看控制器状态，防止伺服掉使能或控制模式退出后继续发送动作。
        sdk_state = self.sdk_ready_state()
        if sdk_state is False:
            self._ready_confirmed = False
            self._collision_confirmed = False
            raise CarmCommandError(
                f"{action_name} is blocked because the controller no longer "
                "reports a ready POSITION state."
            )

    def _require_motion_ready(self) -> None:
        """完成 _require_motion_ready 对应的内部校验或数据转换。"""

        self._require_action_ready("motion")
        if (
            self.config.collision_required_for_motion
            and not self.collision_confirmed
        ):
            raise CarmCommandError(
                "Motion is blocked because collision.required_for_motion=true "
                "but collision configuration was not confirmed in this connection."
            )

    def _validate_workspace_xyz(self, x: float, y: float, z: float) -> None:
        """完成 _validate_workspace_xyz 对应的内部校验或数据转换。"""

        assert_finite((x, y, z), "target xyz")
        checks = (
            ("x", x, self.config.workspace_x_min_m, self.config.workspace_x_max_m),
            ("y", y, self.config.workspace_y_min_m, self.config.workspace_y_max_m),
            ("z", z, self.config.workspace_z_min_m, self.config.workspace_z_max_m),
        )
        for axis, value, lower, upper in checks:
            if not lower <= value <= upper:
                raise CarmCommandError(
                    f"Target {axis}={value:.4f} m is outside configured workspace "
                    f"[{lower:.4f}, {upper:.4f}] m."
                )

    def _run_action(self, name: str, call: Callable[[], Any]) -> Any:
        """执行一条状态改变命令；异常或拒绝时尽力停止。"""

        self._action_started = True
        try:
            return self._call(name, call())
        except Exception:
            self.stop_best_effort()
            raise

    def _run_motion_action(
        self,
        name: str,
        call: Callable[[], Any],
        *,
        target_pose: PoseXYZQuat | Sequence[float],
    ) -> Any:
        """给 SDK 的超长同步等待增加包装层 watchdog。

        当前 CArm SDK 的同步运动内部等待上限接近无限。本方法在守护线程中调用
        SDK；超过 ``motion_timeout_s`` 时主线程立即尽力停止任务并报错。Python
        无法强制杀死阻塞线程，因此现场仍必须验证 stop 行为并保留硬件急停。
        即使 SDK 返回“接受/成功”，也必须重新读取 configured TCP，并在明确
        的位置、姿态容差内确认实际到达后才向调用方报告成功。
        """

        self._action_started = True
        target = PoseXYZQuat.from_sequence(self._pose_list(target_pose))
        result: dict[str, Any] = {}
        error: dict[str, BaseException] = {}

        def invoke() -> None:
            """完成 CarmClient.invoke 对应的操作。"""

            try:
                result["response"] = call()
            except BaseException as exc:  # 在线程边界保存后由主线程重新抛出。
                error["exception"] = exc

        worker = threading.Thread(
            target=invoke,
            name=f"hps-carm-{name}",
            daemon=True,
        )
        worker.start()
        worker.join(self.config.motion_timeout_s)
        if worker.is_alive():
            self.stop_best_effort()
            worker.join(self.config.motion_stop_grace_s)
            raise CarmCommandError(
                f"{name} exceeded motion timeout "
                f"{self.config.motion_timeout_s:.3f} s; stop was requested. "
                "Keep the hardware emergency stop available."
            )
        if "exception" in error:
            self.stop_best_effort()
            raise error["exception"]
        try:
            response = self._call(name, result.get("response"))
            self._verify_motion_completion(name, target)
            return response
        except Exception:
            self.stop_best_effort()
            raise

    def _verify_motion_completion(
        self,
        name: str,
        target: PoseXYZQuat,
    ) -> None:
        """用 configured TCP 实际位姿验证同步运动真的完成。"""

        if not self.is_connected():
            self._ready_confirmed = False
            self._collision_confirmed = False
            raise CarmCommandError(
                f"{name} was accepted, but CArm disconnected before completion "
                "could be verified."
            )
        if not self._ready_confirmed:
            raise CarmCommandError(
                f"{name} was accepted, but the local ready gate was lost before "
                "completion could be verified."
            )
        sdk_state = self.sdk_ready_state()
        if sdk_state is False:
            self._ready_confirmed = False
            self._collision_confirmed = False
            raise CarmCommandError(
                f"{name} was accepted, but the controller no longer reports a "
                "ready POSITION state."
            )

        actual = self.current_tool_pose()
        position_error_m = math.dist(
            (actual.x, actual.y, actual.z),
            (target.x, target.y, target.z),
        )
        rotation_error_deg = math.degrees(
            quat_angular_distance_rad(
                (actual.qx, actual.qy, actual.qz, actual.qw),
                (target.qx, target.qy, target.qz, target.qw),
            )
        )
        if (
            position_error_m
            > self.config.motion_completion_position_tolerance_m + 1e-12
            or rotation_error_deg
            > self.config.motion_completion_rotation_tolerance_deg + 1e-9
        ):
            raise CarmCommandError(
                f"{name} was accepted but configured TCP did not reach the target: "
                f"position_error={position_error_m:.6f} m "
                f"(tolerance={self.config.motion_completion_position_tolerance_m:.6f} m), "
                f"rotation_error={rotation_error_deg:.3f} deg "
                f"(tolerance="
                f"{self.config.motion_completion_rotation_tolerance_deg:.3f} deg)."
            )

    def _pose_list(self, pose: PoseXYZQuat | Sequence[float]) -> list[float]:
        """完成 _pose_list 对应的内部校验或数据转换。"""

        if isinstance(pose, PoseXYZQuat):
            pose_list = pose.as_list()
        else:
            pose_list = []
            for value in pose:
                pose_list.append(float(value))
        if len(pose_list) != 7:
            raise ValueError("CArm pose must be [x, y, z, qx, qy, qz, qw].")
        assert_finite(pose_list, "pose")
        pose_list[3:] = list(normalize_quat(pose_list[3:]))
        return pose_list

    def _call(self, name: str, response: Any) -> Any:
        # 不同 SDK 版本返回值可能略有差异，统一在这里判断是否成功。
        """完成 _call 对应的内部校验或数据转换。"""

        if not self.config.require_accepted_response:
            return response
        if self._response_ok(response):
            return response
        raise CarmCommandError(f"{name} failed or was rejected: {response!r}")

    def _response_ok(self, response: Any) -> bool:
        """完成 _response_ok 对应的内部校验或数据转换。"""

        if response is None:
            # require_accepted_response=true 时必须收到明确成功值；None 不能证明
            # 控制器接受了命令。本包装器把该门禁设为强制项，不允许通过配置
            # 关闭；返回值语义不同的 SDK 必须先单独适配，不能在这里静默放行。
            return False
        if isinstance(response, bool):
            return response
        if isinstance(response, str):
            return response in self.config.accepted_response_values
        if isinstance(response, Mapping):
            explicit = (
                response.get("recv")
                or response.get("status")
                or response.get("result")
            )
            if isinstance(explicit, str):
                normalized = explicit.strip().lower()
                response_is_failure = False
                for marker in ("refuse", "reject", "error", "fail"):
                    if marker in normalized:
                        response_is_failure = True
                        break
                if response_is_failure:
                    return False
                if explicit in self.config.accepted_response_values:
                    return True
                # 存在未知显式状态时，不允许 code=0 等旁路把它覆盖成成功。
                return False
            if response.get("success") is False:
                return False
            if response.get("success") is True:
                return True
            if (
                type(response.get("code")) is int
                and response.get("code") == 0
            ) or (
                type(response.get("errCode")) is int
                and response.get("errCode") == 0
            ):
                return True
            return False
        # 未知对象、列表和非零数字都不是明确成功证据。
        return False

"""试管抓取与历史试管架孔位使用的二维几何函数。"""


import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


# 本项目采集分辨率为 1280×720。小于 30 px 的实例通常不足以提供稳定端点，
# PCA 长短轴比低于 2.0 的实例则太接近圆形/方形，方向容易随像素噪声跳变。
# 调用方可在经过现场验证后通过 tube_pose_from_masks 的关键字参数收紧阈值。
DEFAULT_MIN_TUBE_LENGTH_PX = 30.0
DEFAULT_MIN_TUBE_PCA_ASPECT_RATIO = 2.0


@dataclass(frozen=True)
class TubePose2D:
    """保存一支试管在图像中的 B、C、G 点、长轴和质量指标。"""

    center_xy: tuple[float, float]
    bottom_xy: tuple[float, float]
    cap_xy: tuple[float, float]
    grasp_xy: tuple[float, float]
    axis_xy: tuple[float, float]
    angle_rad: float
    # 兼容旧的手工构造调用；由 tube_pose_from_masks 产生的结果始终填写这两项。
    length_px: float | None = None
    pca_aspect_ratio: float | None = None


@dataclass(frozen=True)
class RackGrid:
    """保存历史试管架网格的坐标轴、孔位中心和外框。"""

    origin_xy: tuple[float, float]
    x_axis: tuple[float, float]
    y_axis: tuple[float, float]
    centers_xy: list[list[tuple[float, float]]]
    box_xy: list[tuple[float, float]]


def mask_points_xy(mask: np.ndarray) -> np.ndarray:
    """把二值掩膜前景像素转换为 N×2 的 [x, y] 点数组。"""

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("Mask must be a 2D array.")
    ys, xs = np.nonzero(array.astype(bool))
    if len(xs) == 0:
        raise ValueError("Mask has no foreground pixels.")
    return np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])


def largest_connected_component(
    mask: np.ndarray,
    *,
    name: str,
    min_main_fraction: float = 0.85,
) -> np.ndarray:
    """保留最大连通域，并拒绝由多个大块拼成的可疑实例 mask。"""

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name} mask must be a 2D array.")
    try:
        finite = np.all(np.isfinite(array))
    except TypeError as exc:
        raise ValueError(f"{name} mask must contain numeric values.") from exc
    if not finite:
        raise ValueError(f"{name} mask contains NaN or infinity.")
    binary = array.astype(bool).astype(np.uint8)
    foreground = int(binary.sum())
    if foreground == 0:
        raise ValueError(f"{name} mask has no foreground pixels.")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    if count <= 1:
        raise ValueError(f"{name} mask has no foreground component.")
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(component_areas)) + 1
    largest_area = int(component_areas[largest_label - 1])
    main_fraction = largest_area / foreground
    if main_fraction < min_main_fraction:
        raise ValueError(
            f"{name} mask contains multiple substantial components: "
            f"largest fraction={main_fraction:.3f}."
        )
    return labels == largest_label


def pca_axis(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # PCA 的第一主方向就是试管 mask 的长轴方向。
    """通过 PCA 计算点云的主轴方向和长短轴尺度。"""

    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("PCA points must have shape (N, 2).")
    if len(points_xy) < 2:
        raise ValueError("At least 2 foreground pixels are required for PCA.")
    if not np.all(np.isfinite(points_xy)):
        raise ValueError("PCA points contain NaN or infinity.")
    center = points_xy.mean(axis=0)
    centered = points_xy - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise ValueError("Mask has no stable principal axis.")
    axis = axis / norm
    return center, axis


def axis_endpoints(points_xy: np.ndarray, center_xy: np.ndarray, axis_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 把所有 mask 点投影到长轴上，最小/最大投影就是试管两端。
    """根据主轴投影分位数计算物体两端点。"""

    projection = (points_xy - center_xy) @ axis_xy
    return center_xy + projection.min() * axis_xy, center_xy + projection.max() * axis_xy


def tube_pose_from_masks(
    tube_mask: np.ndarray,
    cap_mask: np.ndarray | None = None,
    grasp_fraction_from_bottom: float = 0.5,
    *,
    min_tube_length_px: float = DEFAULT_MIN_TUBE_LENGTH_PX,
    min_tube_pca_aspect_ratio: float = DEFAULT_MIN_TUBE_PCA_ASPECT_RATIO,
) -> TubePose2D:
    """计算试管中心、B/C 端点、抓取点和图像角度。

    管盖掩膜用于消除长轴 180° 歧义，返回方向从无盖端 B 指向管盖端 C。
    """

    if isinstance(grasp_fraction_from_bottom, bool):
        raise ValueError(
            "grasp_fraction_from_bottom must be numeric, not a boolean."
        )
    try:
        grasp_fraction = float(grasp_fraction_from_bottom)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "grasp_fraction_from_bottom must be a numeric value in [0, 1]."
        ) from exc
    if not math.isfinite(grasp_fraction) or not 0.0 <= grasp_fraction <= 1.0:
        raise ValueError(
            "grasp_fraction_from_bottom must be finite and within [0, 1]."
        )
    if isinstance(min_tube_length_px, bool) or isinstance(
        min_tube_pca_aspect_ratio, bool
    ):
        raise ValueError(
            "Tube geometry quality thresholds must be numeric, not booleans."
        )
    try:
        min_length = float(min_tube_length_px)
        min_aspect = float(min_tube_pca_aspect_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Tube geometry quality thresholds must be numeric values."
        ) from exc
    if not math.isfinite(min_length) or min_length <= 0.0:
        raise ValueError("min_tube_length_px must be finite and greater than zero.")
    if not math.isfinite(min_aspect) or min_aspect <= 1.0:
        raise ValueError(
            "min_tube_pca_aspect_ratio must be finite and greater than 1.0."
        )

    tube_array = np.asarray(tube_mask)
    if cap_mask is not None:
        cap_array = np.asarray(cap_mask)
        if cap_array.shape != tube_array.shape:
            raise ValueError(
                "Cap mask shape must match the tube mask shape: "
                f"cap={cap_array.shape}, tube={tube_array.shape}."
            )
        if not np.any(cap_array):
            raise ValueError(
                "A cap mask was provided but contains no foreground pixels; "
                "tube direction cannot be resolved."
            )
    else:
        cap_array = None

    clean_tube_mask = largest_connected_component(
        tube_array,
        name="Tube",
    )
    points = mask_points_xy(clean_tube_mask)
    if len(points) < 10:
        raise ValueError(
            f"Tube mask is too small for stable geometry: {len(points)} pixels."
        )
    # 先做一次 PCA，再按长轴投影去掉两端极少量离群像素，随后重新拟合。
    # 这能抑制分割 mask 中孤立小岛或细长毛刺对抓取中心和端点的拖拽。
    initial_center, initial_axis = pca_axis(points)
    initial_projection = (points - initial_center) @ initial_axis
    lower, upper = np.percentile(initial_projection, (5.0, 95.0))
    core_points = points[
        (initial_projection >= lower) & (initial_projection <= upper)
    ]
    if len(core_points) < 10:
        raise ValueError(
            "Tube mask has too few robust core pixels after outlier filtering."
        )
    center, axis = pca_axis(core_points)
    end_a, end_b = axis_endpoints(core_points, center, axis)
    tube_length_px = float(np.linalg.norm(end_b - end_a))
    if not math.isfinite(tube_length_px) or tube_length_px < min_length:
        raise ValueError(
            "Tube mask principal-axis length is too small: "
            f"{tube_length_px:.3f} px < {min_length:.3f} px."
        )

    # 试管管身应明显细长。接近圆形/方形的 mask 长轴会随像素噪声跳动。
    centered = core_points - center
    singular_values = np.linalg.svd(centered, compute_uv=False)
    minor = 0.0
    if len(singular_values) > 1:
        minor = float(singular_values[1])
    aspect_ratio = float(singular_values[0]) / max(minor, 1e-12)
    if not math.isfinite(aspect_ratio) or aspect_ratio < min_aspect:
        raise ValueError(
            "Tube mask is not elongated enough for a stable long axis: "
            f"PCA aspect ratio={aspect_ratio:.3f} < {min_aspect:.3f}."
        )

    if cap_array is not None:
        # cap mask 用来判断哪一端是盖子，解决长轴 180 度方向不确定的问题。
        clean_cap_mask = largest_connected_component(
            cap_array,
            name="Cap",
            min_main_fraction=0.75,
        )
        cap_points = mask_points_xy(clean_cap_mask)
        if len(cap_points) < 3:
            raise ValueError(
                f"Cap mask is too small for pairing: {len(cap_points)} pixels."
            )
        cap_center = cap_points.mean(axis=0)

        # 最高分 body/cap 可能来自不同实例。要求盖子中心靠近管身长轴及某一端，
        # 否则拒绝输出方向，而不是把错误配对转换成机械臂 yaw。
        perpendicular = np.array([-axis[1], axis[0]], dtype=np.float64)
        half_width_px = float(
            np.percentile(np.abs(centered @ perpendicular), 95)
        )
        cap_perpendicular_px = abs(float((cap_center - center) @ perpendicular))
        max_perpendicular_px = max(3.0 * half_width_px, 0.15 * tube_length_px, 3.0)
        distance_to_a_px = float(np.linalg.norm(cap_center - end_a))
        distance_to_b_px = float(np.linalg.norm(cap_center - end_b))
        nearest_endpoint_px = min(distance_to_a_px, distance_to_b_px)
        if cap_perpendicular_px > max_perpendicular_px:
            raise ValueError(
                "Cap/body pairing rejected: cap is too far from the tube axis "
                f"({cap_perpendicular_px:.2f} px > {max_perpendicular_px:.2f} px)."
            )
        if nearest_endpoint_px > 0.35 * tube_length_px:
            raise ValueError(
                "Cap/body pairing rejected: cap is too far from both tube ends "
                f"({nearest_endpoint_px:.2f} px)."
            )
        endpoint_margin_px = abs(distance_to_a_px - distance_to_b_px)
        if endpoint_margin_px < 0.20 * tube_length_px:
            raise ValueError(
                "Cap/body pairing rejected: cap does not identify one tube end "
                f"unambiguously (distance margin={endpoint_margin_px:.2f} px)."
            )
        cap_is_a = distance_to_a_px < distance_to_b_px
        if cap_is_a:
            cap_end = end_a
            bottom_end = end_b
        else:
            cap_end = end_b
            bottom_end = end_a
    else:
        # 没有 cap 时只能知道角度，不能稳定知道头尾方向。
        cap_end = end_b
        bottom_end = end_a

    axis_bottom_to_cap = cap_end - bottom_end
    directed_length = float(np.linalg.norm(axis_bottom_to_cap))
    if not math.isfinite(directed_length) or directed_length < 1e-9:
        raise ValueError("Tube endpoints collapse to a zero-length axis.")
    axis_bottom_to_cap = axis_bottom_to_cap / directed_length

    # grasp_fraction=0.5 表示夹在试管中点；也可以向底部或盖子方向微调。
    grasp = bottom_end + grasp_fraction * (cap_end - bottom_end)
    angle = math.atan2(axis_bottom_to_cap[1], axis_bottom_to_cap[0])

    return TubePose2D(
        center_xy=tuple(center.tolist()),
        bottom_xy=tuple(bottom_end.tolist()),
        cap_xy=tuple(cap_end.tolist()),
        grasp_xy=tuple(grasp.tolist()),
        axis_xy=tuple(axis_bottom_to_cap.tolist()),
        angle_rad=float(angle),
        length_px=tube_length_px,
        pca_aspect_ratio=aspect_ratio,
    )


def largest_contour(mask: np.ndarray) -> np.ndarray:
    """返回掩膜中面积最大的 OpenCV 轮廓。"""

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Mask has no contour.")
    return max(contours, key=cv2.contourArea)


def order_box_points(points_xy: np.ndarray) -> np.ndarray:
    """把旋转矩形四个角按稳定顺序排列。"""

    points = points_xy.astype(np.float64)
    sums = points.sum(axis=1)
    diffs = points[:, 0] - points[:, 1]
    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmax(diffs)]
    bottom_left = points[np.argmin(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float64)


def unit(vector_xy: np.ndarray) -> np.ndarray:
    """把二维向量归一化为单位向量。"""

    norm = np.linalg.norm(vector_xy)
    if norm < 1e-9:
        raise ValueError("Cannot normalize a zero vector.")
    return vector_xy / norm


def rack_grid_from_mask(
    rack_mask: np.ndarray,
    rows: int,
    cols: int,
    first_hole_offset_px: Sequence[float],
    pitch_px: Sequence[float],
) -> RackGrid:
    """根据历史试管架掩膜和固定行列参数生成孔位中心。"""

    if isinstance(rows, bool) or isinstance(cols, bool):
        raise ValueError("Rack grid rows and cols must be positive integers.")
    if int(rows) != rows or int(cols) != cols or int(rows) <= 0 or int(cols) <= 0:
        raise ValueError("Rack grid rows and cols must be positive integers.")
    try:
        offsets = np.asarray(first_hole_offset_px, dtype=np.float64)
        pitches = np.asarray(pitch_px, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Rack grid offset and pitch must be numeric XY pairs.") from exc
    if offsets.shape != (2,) or pitches.shape != (2,):
        raise ValueError("Rack grid offset and pitch must each contain two values.")
    if not np.all(np.isfinite(offsets)) or not np.all(np.isfinite(pitches)):
        raise ValueError("Rack grid offset and pitch cannot contain NaN or infinity.")
    if np.any(pitches <= 0.0):
        raise ValueError("Rack grid pitch values must be greater than zero.")

    contour = largest_contour(rack_mask)
    if cv2.contourArea(contour) <= 1.0:
        raise ValueError("Rack mask contour is too small for a stable grid.")
    rect = cv2.minAreaRect(contour)

    # minAreaRect 给试管架外接旋转矩形；再按左上、右上、右下、左下排序。
    box = order_box_points(cv2.boxPoints(rect))
    top_left, top_right, _, bottom_left = box

    # x_axis/y_axis 是试管架自己的坐标轴，即使图像里架子轻微旋转也能跟着转。
    x_axis = unit(top_right - top_left)
    y_axis = unit(bottom_left - top_left)
    origin = top_left
    offset_x, offset_y = offsets.tolist()
    pitch_x, pitch_y = pitches.tolist()
    first = origin + offset_x * x_axis + offset_y * y_axis

    centers: list[list[tuple[float, float]]] = []
    for row in range(int(rows)):
        row_centers: list[tuple[float, float]] = []
        for col in range(int(cols)):
            # 第一个孔位置 + 行列间距 = 每一个孔中心。
            point = first + col * pitch_x * x_axis + row * pitch_y * y_axis
            row_centers.append(tuple(point.tolist()))
        centers.append(row_centers)

    box_points: list[tuple[float, float]] = []
    for point in box:
        box_points.append(tuple(point.tolist()))

    return RackGrid(
        origin_xy=tuple(origin.tolist()),
        x_axis=tuple(x_axis.tolist()),
        y_axis=tuple(y_axis.tolist()),
        centers_xy=centers,
        box_xy=box_points,
    )


def draw_tube_pose(image: np.ndarray, pose: TubePose2D) -> np.ndarray:
    """在图像上绘制试管中心、端点、抓取点和长轴。"""

    output = image.copy()
    bottom = tuple(np.round(pose.bottom_xy).astype(int))
    cap = tuple(np.round(pose.cap_xy).astype(int))
    grasp = tuple(np.round(pose.grasp_xy).astype(int))
    cv2.line(output, bottom, cap, (0, 255, 0), 2)
    cv2.circle(output, bottom, 5, (255, 0, 0), -1)
    cv2.circle(output, cap, 5, (0, 0, 255), -1)
    cv2.circle(output, grasp, 5, (0, 255, 255), -1)
    return output


def draw_rack_grid(image: np.ndarray, grid: RackGrid) -> np.ndarray:
    """在图像上绘制历史试管架网格。"""

    output = image.copy()
    box = np.round(np.array(grid.box_xy)).astype(int)
    cv2.polylines(output, [box], isClosed=True, color=(255, 255, 0), thickness=2)
    for row in grid.centers_xy:
        for x, y in row:
            cv2.circle(output, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)
    return output

"""当前四类试管实例分割模型的严格类别契约。

三个推理入口（07、08、09）都依赖相同的类别 ID：

    0 -> p-body
    1 -> p-cap
    2 -> y-body
    3 -> y-cap

只检查“名称集合中包含这些类别”是不够的。类别顺序改变会让后续显示、统计和
几何配对使用错误语义；多出其他类别也通常意味着自动选择到了别的实验权重。
因此这里统一要求模型任务、类别数量、类别 ID 和 ``vision.yaml`` 顺序全部一致。
"""


from collections.abc import Mapping, Sequence
from typing import Any


EXPECTED_TUBE_CLASS_ORDER = ("p-body", "p-cap", "y-body", "y-cap")


def ordered_model_class_names(names: Any) -> tuple[str, ...]:
    """把 Ultralytics ``model.names`` 转成按类别 ID 排列的严格名称元组。"""

    if isinstance(names, Mapping):
        indexed: dict[int, str] = {}
        for raw_class_id, raw_name in names.items():
            if isinstance(raw_class_id, bool):
                raise ValueError("模型类别 ID 不能是布尔值。")
            try:
                class_id = int(raw_class_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"模型类别 ID 不是整数：{raw_class_id!r}。"
                ) from exc
            # 不接受 1.5 -> 1 一类静默截断；字符串 "1" 兼容序列化后的字典键。
            if not (
                isinstance(raw_class_id, str)
                and raw_class_id.strip() == str(class_id)
            ) and class_id != raw_class_id:
                raise ValueError(f"模型类别 ID 不是整数：{raw_class_id!r}。")
            if class_id in indexed:
                raise ValueError(f"模型类别 ID 重复：{class_id}。")
            indexed[class_id] = str(raw_name)

        expected_ids = list(range(len(indexed)))
        if sorted(indexed) != expected_ids:
            raise ValueError(
                "模型类别 ID 必须从 0 开始连续排列："
                f"实际={sorted(indexed)}，期望={expected_ids}。"
            )
        ordered_names: list[str] = []
        for class_id in expected_ids:
            ordered_names.append(indexed[class_id])
        return tuple(ordered_names)

    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        ordered_names = []
        for name in names:
            ordered_names.append(str(name))
        return tuple(ordered_names)
    raise ValueError("模型 names 必须是类别 ID 映射或名称列表。")


def validate_tube_model_contract(
    *,
    task: Any,
    names: Any,
    configured_class_order: Sequence[str],
) -> tuple[str, ...]:
    """严格校验模型任务、模型类别 ID 和 ``vision.yaml`` 类别顺序。

    返回已验证的模型类别顺序，便于调用方打印或测试。函数只检查内存对象，
    不读取相机、不执行推理，也不会访问机械臂。
    """

    if str(task) != "segment":
        raise ValueError(f"模型任务必须是 segment，实际为 {task!r}。")

    configured_names: list[str] = []
    for name in configured_class_order:
        configured_names.append(str(name).strip())
    configured = tuple(configured_names)
    if configured != EXPECTED_TUBE_CLASS_ORDER:
        raise ValueError(
            "vision.yaml 的类别 ID/顺序不符合当前四类模型："
            f"实际={list(configured)}，"
            f"期望={list(EXPECTED_TUBE_CLASS_ORDER)}。"
        )

    actual = ordered_model_class_names(names)
    if actual != configured:
        raise ValueError(
            "模型类别 ID/顺序与 vision.yaml 不一致："
            f"模型={list(actual)}，配置={list(configured)}。"
        )
    return actual


__all__ = [
    "EXPECTED_TUBE_CLASS_ORDER",
    "ordered_model_class_names",
    "validate_tube_model_contract",
]

"""RealSense 实时视觉入口共用的设备身份选择规则。

本模块不在导入时依赖 ``pyrealsense2``。调用方把已延迟导入的 SDK 模块传入，
因此普通图片推理、单元测试和 ``--help`` 不会因为未连接相机而失败。
"""


from typing import Any, Sequence


def choose_realsense_serial(
    requested_serial: str | None,
    available_serials: Sequence[str | None],
) -> str:
    """选择唯一设备序列号；多设备、未知身份和错误显式选择一律拒绝。"""

    requested: str | None = None
    if isinstance(requested_serial, str) and requested_serial.strip():
        requested = requested_serial.strip()
    if requested_serial is not None and requested is None:
        raise ValueError("RealSense --serial 不能是空字符串。")

    normalized: list[str | None] = []
    for serial in available_serials:
        if isinstance(serial, str) and serial.strip():
            normalized.append(serial.strip())
        else:
            normalized.append(None)
    if requested is not None:
        if requested not in normalized:
            visible: list[str] = []
            for serial in normalized:
                if serial is not None:
                    visible.append(serial)
            raise ValueError(
                f"--serial={requested!r} 不在当前 RealSense 设备中：{visible}"
            )
        return requested

    if not normalized:
        raise RuntimeError("没有检测到 RealSense 设备。")
    if len(normalized) > 1:
        visible = []
        for serial in normalized:
            if serial is None:
                visible.append("<序列号不可用>")
            else:
                visible.append(serial)
        raise ValueError(
            "检测到多台 RealSense，必须显式提供 --serial，避免选错相机："
            f"{visible}"
        )
    if normalized[0] is None:
        raise RuntimeError(
            "唯一 RealSense 无法读取序列号，不能建立可追溯的视频源身份。"
        )
    return normalized[0]


def select_realsense_device_serial(
    rs_module: Any,
    requested_serial: str | None,
) -> str:
    """枚举 SDK 设备并按严格规则返回要传给 ``enable_device`` 的序列号。"""

    devices = list(rs_module.context().query_devices())
    serials: list[str | None] = []
    for device in devices:
        try:
            key = rs_module.camera_info.serial_number
            if hasattr(device, "supports") and not device.supports(key):
                serials.append(None)
            else:
                serials.append(str(device.get_info(key)))
        except Exception:
            # 无法读取身份时宁可阻止视频源，也不能让 SDK 随机选择。
            serials.append(None)
    return choose_realsense_serial(requested_serial, serials)


# =============================================================================
# 1. 本次实验真正需要调整的参数
# =============================================================================

CALIBRATION_DIR = Path("/home/gaoyuan/camera_hand_calibration/config")
DEFAULT_IP = "10.42.0.101"
TOOL_INDEX = 1

IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_FPS = 1280, 720, 30
INFERENCE_SIZE, CONFIDENCE, IOU, DEVICE = 1024, 0.25, 0.70, "0"

GRASP_Z_M = 0.165
LIFT_Z_M = 0.250
WAIT_SECONDS = 1.5

# 夹爪向下且参考 yaw=0 的四元数，沿用已通过脚本 11/12 检查的值。
DOWN_QUATERNION = (0.999575504, 0.008135427, 0.027844000, 0.002709061)
READY_JOINTS = (-0.001726, 1.751210, -0.626573, -0.000954, 0.446518, -0.000954)
ZERO_JOINTS = np.zeros(6, dtype=float)

STABLE_FRAMES = 10
MAX_CENTER_JITTER_PX = 3.0
MAX_ANGLE_JITTER_DEG = 3.0


# =============================================================================
# 2. 模型、相机和黄色试管几何
# =============================================================================

def latest_model() -> str:
    """省略 --model 时，使用 runs 中最新的 best.pt。"""

    candidates: list[Path] = []
    for path in (ROOT / "runs").rglob("best.pt"):
        if path.is_file():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("runs 下没有 best.pt，请用 --model 指定权重")
    latest_path = candidates[0]
    latest_time = latest_path.stat().st_mtime
    for path in candidates[1:]:
        modified_time = path.stat().st_mtime
        if modified_time > latest_time:
            latest_path = path
            latest_time = modified_time
    return str(latest_path.resolve())


class RealSenseColor:
    """只打开一台明确的 RealSense，并读取彩色图。"""

    def __init__(self, serial: str | None):
        """初始化当前对象，并保存后续操作需要的状态。"""

        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        self.started = False
        self.serial = select_realsense_device_serial(rs, serial)

        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.color,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            rs.format.bgr8,
            CAMERA_FPS,
        )
        profile = self.pipeline.start(config)
        self.started = True

        device = profile.get_device()
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = stream.get_intrinsics()
        self.intrinsics = np.array(
            [intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy],
            dtype=float,
        )
        name = device.get_info(rs.camera_info.name)
        print(
            f"相机：{name} serial={self.serial} "
            f"color={stream.width()}x{stream.height()}@{stream.fps()}"
        )

    def read(self) -> np.ndarray | None:
        """读取一帧数据；暂时没有有效帧时返回 None。"""

        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        frame = frames.get_color_frame()
        if frame:
            return np.asanyarray(frame.get_data())
        return None

    def close(self) -> None:
        """释放当前对象占用的相机、文件或连接资源。"""

        if self.started:
            self.pipeline.stop()
            self.started = False


def yellow_geometry(result, image_shape) -> tuple[np.ndarray, ...] | None:
    """从一帧结果中取唯一的 y-body/y-cap，并返回中心、B、C。"""

    if result.boxes is None or result.masks is None or len(result.boxes) == 0:
        return None

    ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    body = np.flatnonzero(ids == 2)
    cap = np.flatnonzero(ids == 3)
    if len(body) != 1 or len(cap) != 1:
        return None

    height, width = image_shape
    masks = result.masks.data.detach().cpu().numpy()

    def resize(index: int) -> np.ndarray:
        """完成 resize 对应的单一处理步骤。"""

        return cv2.resize(
            masks[index].astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        ) > 0.5

    try:
        pose = tube_pose_from_masks(resize(int(body[0])), resize(int(cap[0])))
    except ValueError:
        return None
    return (
        np.asarray(pose.center_xy),
        np.asarray(pose.bottom_xy),
        np.asarray(pose.cap_xy),
    )


def stable_geometry(history) -> tuple[np.ndarray, ...] | None:
    """中心和有方向的 B→C 角度连续稳定 10 帧后才允许锁定。"""

    if len(history) < STABLE_FRAMES:
        return None
    center_values: list[np.ndarray] = []
    bottom_values: list[np.ndarray] = []
    cap_values: list[np.ndarray] = []
    for item in history:
        center_values.append(item[0])
        bottom_values.append(item[1])
        cap_values.append(item[2])
    centers = np.asarray(center_values)
    bottoms = np.asarray(bottom_values)
    caps = np.asarray(cap_values)

    center = np.median(centers, axis=0)
    if np.max(np.linalg.norm(centers - center, axis=1)) > MAX_CENTER_JITTER_PX:
        return None

    angles = np.arctan2((caps - bottoms)[:, 1], (caps - bottoms)[:, 0])
    mean = math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
    errors: list[float] = []
    for angle in angles:
        errors.append(abs(wrap_pi(float(angle) - mean)))
    if math.degrees(max(errors)) > MAX_ANGLE_JITTER_DEG:
        return None
    return center, np.median(bottoms, axis=0), np.median(caps, axis=0)


# =============================================================================
# 3. 手眼转换：像素中心和 B→C -> 基座 XY/yaw
# =============================================================================

def load_calibration() -> dict:
    """读取现有最终手眼矩阵，并检查三份变换是否一致。"""

    def matrix(name: str, shape) -> np.ndarray:
        """读取一个矩阵文件，并检查尺寸和有限值。"""

        value = np.loadtxt(CALIBRATION_DIR / name, delimiter=",", dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name} 不是有效的 {shape} 矩阵")
        return value

    with (CALIBRATION_DIR / "camera.yaml").open("r", encoding="utf-8") as file:
        camera = yaml.safe_load(file)
    intrinsic = matrix("intrinsic.txt", (3, 3))
    camera_from_workspace = matrix("T_cam2ws.txt", (4, 4))
    arm_from_workspace = matrix("T_arm2ws.txt", (4, 4))
    arm_from_camera = matrix("T_arm2cam.txt", (4, 4))

    expected = arm_from_workspace @ np.linalg.inv(camera_from_workspace)
    error = float(np.max(np.abs(expected - arm_from_camera)))
    if error > 1e-3:
        raise ValueError(f"手眼矩阵组合不一致：最大误差 {error:.6f}")
    return {
        "intrinsic": intrinsic,
        "camera_from_workspace": camera_from_workspace,
        "arm_from_workspace": arm_from_workspace,
        "width": int(camera["img_width"]),
        "height": int(camera["img_height"]),
        "matrix_error": error,
    }


def pixel_to_arm(point, calibration) -> np.ndarray:
    """把彩色图像像素投影到标定工作台，再转到机械臂基座，单位 m。"""

    u, v = np.asarray(point, dtype=float)
    if not (0 <= u < calibration["width"] and 0 <= v < calibration["height"]):
        raise ValueError(f"像素 ({u:.1f}, {v:.1f}) 超出标定图像")

    ray_camera = np.linalg.solve(
        calibration["intrinsic"],
        np.array([u, v, 1.0]),
    )
    workspace_from_camera = np.linalg.inv(calibration["camera_from_workspace"])
    origin = workspace_from_camera[:3, 3]
    ray_workspace = workspace_from_camera[:3, :3] @ ray_camera
    scale = -origin[2] / ray_workspace[2]
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("像素射线不能与工作台正向相交")

    workspace_mm = origin + scale * ray_workspace
    workspace_mm[2] = 0.0
    arm_mm = (
        calibration["arm_from_workspace"] @ np.append(workspace_mm, 1.0)
    )[:3]
    return arm_mm / 1000.0


def wrap_pi(angle: float) -> float:
    """把弧度限制到 [-pi, pi)。"""

    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def make_solution(geometry, calibration, yaw_offset_deg: float) -> dict:
    """计算基座中心、试管 yaw，并选离参考 yaw=0 更近的等价夹爪方向。"""

    center, bottom, cap = geometry
    center_arm = pixel_to_arm(center, calibration)
    bottom_arm = pixel_to_arm(bottom, calibration)
    cap_arm = pixel_to_arm(cap, calibration)
    axis = cap_arm[:2] - bottom_arm[:2]
    if np.linalg.norm(axis) < 1e-6:
        raise ValueError("B/C 转换后距离太短，不能计算 yaw")

    tube_yaw = math.atan2(float(axis[1]), float(axis[0]))
    yaw_1 = wrap_pi(tube_yaw + math.radians(yaw_offset_deg))
    yaw_2 = wrap_pi(yaw_1 + math.pi)
    gripper_yaw = min((yaw_1, yaw_2), key=abs)
    other_yaw = yaw_1
    if gripper_yaw == yaw_1:
        other_yaw = yaw_2
    return {
        "center": center,
        "bottom": bottom,
        "cap": cap,
        "center_arm": center_arm,
        "tube_yaw": tube_yaw,
        "gripper_yaw": gripper_yaw,
        "other_yaw": other_yaw,
        "yaw_offset_deg": yaw_offset_deg,
    }


def check_live_intrinsics(source: RealSenseColor, calibration: dict) -> None:
    """比较实时 RealSense 内参与手眼标定内参。"""

    saved = calibration["intrinsic"]
    expected = np.array([saved[0, 0], saved[1, 1], saved[0, 2], saved[1, 2]])
    error = float(np.max(np.abs(source.intrinsics - expected)))
    if error > 1.0:
        raise ValueError(f"实时相机内参与标定不一致：最大差值 {error:.3f} px")
    print(f"相机内参检查通过：最大差值 {error:.3f} px")


# =============================================================================
# 4. 实时检测与机械臂动作
# =============================================================================

def detect_and_lock(args, calibration) -> dict:
    """显示实时结果；稳定后按 C 锁定目标。"""

    from ultralytics import YOLO

    model_path = args.model or latest_model()
    model = YOLO(model_path)
    validate_tube_model_contract(
        task=model.task,
        names=model.names,
        configured_class_order=EXPECTED_TUBE_CLASS_ORDER,
    )
    print(f"模型：{model_path}")

    source = RealSenseColor(args.serial)
    history = deque(maxlen=STABLE_FRAMES)
    window = "13 - yellow tube yaw grasp"
    try:
        check_live_intrinsics(source, calibration)
        print("目标稳定后按 C 锁定；按 Q 或 Esc 退出。")
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        while True:
            frame = source.read()
            if frame is None:
                continue
            result = model.predict(
                frame,
                imgsz=INFERENCE_SIZE,
                conf=CONFIDENCE,
                iou=IOU,
                device=DEVICE,
                # 项目视觉脚本统一固定使用 FP32。
                half=False,
                retina_masks=True,
                verbose=False,
            )[0]
            geometry = yellow_geometry(result, frame.shape[:2])
            if geometry is not None:
                history.append(geometry)
            else:
                history.clear()
            stable = stable_geometry(history)
            solution = None
            if stable is not None:
                solution = make_solution(
                    stable, calibration, args.yaw_offset_deg
                )

            shown = result.plot(masks=True, boxes=True, labels=True, conf=True)
            if geometry is not None:
                rounded_points: list[tuple[int, int]] = []
                for point in geometry:
                    rounded_points.append(tuple(np.round(point).astype(int)))
                center, bottom, cap = rounded_points
                cv2.line(shown, bottom, cap, (0, 255, 255), 3)
                cv2.circle(shown, center, 7, (0, 255, 0), -1)
            text = "Waiting for stable y-body/y-cap"
            text_color = (0, 165, 255)
            if solution:
                text = "STABLE - C: lock"
                text_color = (0, 255, 0)
            cv2.putText(
                shown,
                text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                text_color,
                2,
            )
            if solution:
                cv2.putText(
                    shown,
                    f"BASE yaw={math.degrees(solution['tube_yaw']):.1f}  "
                    f"GRIPPER yaw={math.degrees(solution['gripper_yaw']):.1f}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            cv2.imshow(window, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                raise KeyboardInterrupt
            if key in (ord("c"), ord("C")) and solution:
                return solution
    finally:
        source.close()
        cv2.destroyAllWindows()


def print_plan(solution) -> None:
    """把即将检查或执行的机械臂计划打印到终端。"""

    x, y, _ = solution["center_arm"]
    print("\n========== 动态 yaw 抓取计划 ==========")
    print(f"黄色中心基座 XY：({x:.6f}, {y:.6f}) m")
    print(f"试管基座 B→C yaw：{math.degrees(solution['tube_yaw']):.3f}°")
    print(f"yaw 偏移：{solution['yaw_offset_deg']:.3f}°")
    print(f"优先候选 yaw：{math.degrees(solution['gripper_yaw']):.3f}°")
    print(f"另一等价 yaw：{math.degrees(solution['other_yaw']):.3f}°")
    print(f"抓取/抬高 Z：{GRASP_Z_M:.3f} / {LIFT_Z_M:.3f} m")
    print(
        "动作：零位检查 -> 准备位 -> 高位对齐并开爪 -> 保持 yaw 下降"
        " -> 等待 3 秒 -> 闭爪 -> 保持 yaw 抬高并停止"
    )
    print("不会转竖直、不会松爪、不会回零，也不会宣称传感器已确认抓取成功。")
    print("========================================\n")


def choose_yaw_by_ik(robot: CarmClient, solution) -> tuple[float, list[float]]:
    """对两个等价 yaw 求 IK，选择从准备位关节变化较小的解。"""

    current = np.asarray(robot.arm.joint_pos, dtype=float)
    choices = []
    for yaw in (solution["gripper_yaw"], solution["other_yaw"]):
        pose = robot.make_xyzyaw_pose(
            solution["center_arm"][0],
            solution["center_arm"][1],
            LIFT_Z_M,
            yaw,
        )
        robot.validate_pose_target(pose, check_step=False)
        response = robot.arm.inverse_kine(
            pose.as_list(),
            current.tolist(),
            tool=TOOL_INDEX,
        )
        try:
            joints = np.asarray(response["data"]["joint1"], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        if joints.shape != (6,) or not np.isfinite(joints).all():
            continue
        fk = np.asarray(
            robot.arm.forward_kine(joints.tolist(), tool=TOOL_INDEX),
            dtype=float,
        )
        if fk.shape != (7,) or not np.isfinite(fk).all():
            continue
        position_error_m = float(np.linalg.norm(fk[:3] - pose.as_list()[:3]))
        rotation_error_deg = math.degrees(
            quat_angular_distance_rad(fk[3:7], pose.as_list()[3:7])
        )
        if position_error_m > 0.002 or rotation_error_deg > 1.0:
            continue
        change_deg = math.degrees(float(np.max(np.abs(joints - current))))
        if change_deg <= 95.0:
            choices.append((change_deg, yaw, joints.tolist()))

    if not choices:
        raise RuntimeError("两个等价 yaw 都没有通过 IK/关节变化检查")
    selected_choice = choices[0]
    for choice in choices[1:]:
        if choice[0] < selected_choice[0]:
            selected_choice = choice
    change_deg, yaw, joints = selected_choice
    print(
        f"IK 选用夹爪 yaw={math.degrees(yaw):.3f}°，"
        f"最大关节变化 {change_deg:.2f}°。"
    )
    return yaw, joints


def execute_grasp(solution, args) -> None:
    """执行唯一的 6 步抓取流程，结束时在高位保持闭爪。"""

    x, y, _ = solution["center_arm"]
    confirmation = (
        f"YAW_GRASP_HOLD tool=1 x={x:.3f} y={y:.3f} "
        f"grasp_z={GRASP_Z_M:.3f} lift_z={LIFT_Z_M:.3f}"
    )
    print("请清空路径、确认夹爪中无物体，并准备好急停。")
    if input(f"请输入完整确认文字：\n{confirmation}\n> ").strip() != confirmation:
        print("确认文字不匹配：没有连接机械臂。")
        return

    config = CarmClientConfig(
        ip=args.ip,
        speed_level=1.0,
        tool_index=TOOL_INDEX,
        down_quat_xyzw=DOWN_QUATERNION,
        gripper_yaw_offset_rad=0.0,  # 偏移已经计入 gripper_yaw，不能重复相加。
        gripper_open_m=0.060,
        gripper_close_m=0.000,
        gripper_tau_n=10.0,
        z_safe_m=LIFT_Z_M,
        z_grasp_m=GRASP_Z_M,
        settle_s=1.0,
        setup_verified=True,
        max_single_step_m=0.400,
        max_single_rotation_deg=100.0,
        collision_configure_on_ready=True,
        collision_enabled=True,
        collision_sensitivity_level=2,
        collision_required_for_motion=True,
    )

    with CarmClient(config) as robot:
        joints = np.asarray(robot.arm.joint_pos, dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise RuntimeError("无法读取有效的六关节起始状态")
        zero_error = math.degrees(float(np.max(np.abs(joints - ZERO_JOINTS))))
        print(f"起始零位最大关节误差：{zero_error:.3f}°")
        if zero_error > 3.0:
            raise RuntimeError("机械臂当前不在六关节零位，拒绝开始")

        robot.ready()
        print("1/6：移动到垂直向下准备位。")
        robot.move_joints(READY_JOINTS)

        yaw, above_joints = choose_yaw_by_ik(robot, solution)
        above = robot.make_xyzyaw_pose(x, y, LIFT_Z_M, yaw)
        grasp = robot.make_xyzyaw_pose(x, y, GRASP_Z_M, yaw)
        robot.validate_motion_plan((grasp, above), current_pose=above)

        print("2/6：移动到黄色试管上方并动态对齐 yaw。")
        robot.move_joints(above_joints)
        robot.open_gripper()

        print(f"3/6：保持 yaw 下降到 Z={GRASP_Z_M:.3f} m。")
        robot.move_line_pose(grasp)
        print(f"4/6：等待 {WAIT_SECONDS:.1f} 秒。")
        time.sleep(WAIT_SECONDS)

        print("5/6：闭合夹爪。")
        robot.close_gripper()
        time.sleep(1.0)

        print(f"6/6：保持 yaw 抬高到 Z={LIFT_Z_M:.3f} m。")
        robot.move_line_pose(above)
        print("已到高位并保持闭爪；脚本不会继续转姿、松爪或回零。")


# =============================================================================
# 5. 命令行入口
# =============================================================================

def parse_args():
    """读取终端参数，并返回 argparse.Namespace。"""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="YOLO Seg 权重；默认使用最新 best.pt")
    parser.add_argument("--serial", help="多台 RealSense 时指定序列号")
    parser.add_argument(
        "--yaw-offset-deg",
        type=float,
        default=0.0,
        help="试管方向到夹爪方向的偏移；当前实验默认 0°",
    )
    parser.add_argument("--ip", default=DEFAULT_IP, help="CArm IP")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="确认计划后连接机械臂并执行；默认只预览",
    )
    return parser.parse_args()


def main() -> int:
    """按照本脚本的编号流程依次执行各个步骤。"""

    args = parse_args()
    if not math.isfinite(args.yaw_offset_deg):
        raise ValueError("--yaw-offset-deg 必须是有限数字")

    calibration = load_calibration()
    print(f"手眼矩阵组合检查通过：最大差值 {calibration['matrix_error']:.6f}")
    solution = detect_and_lock(args, calibration)
    print_plan(solution)
    if args.execute:
        execute_grasp(solution, args)
    else:
        print("[PREVIEW ONLY] 没有连接机械臂。确认结果后再添加 --execute。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("\n用户取消。") from None
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"程序终止：{exc}") from None
