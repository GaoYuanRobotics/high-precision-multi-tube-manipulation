#!/usr/bin/env python3
"""第 02 步：用合成数据快速检查项目核心数学计算。

本脚本依次检查二维仿射变换、试管长轴与管盖方向，以及 CArm 成功响应
字符串。所有计算都直接写在本文件中，也不会打开
相机、下载模型或连接机械臂。

任一检查失败会抛出 ``AssertionError``；全部通过时打印成功信息。
初学者建议直接从 ``main()`` 按调用顺序阅读四个检查函数。
"""

from __future__ import annotations

import math

import numpy as np


# =============================================================================
# 1. 小型断言工具
# =============================================================================

def assert_close(name: str, actual: float, expected: float, tolerance=1e-6) -> None:
    """检查两个浮点数的绝对误差，避免直接使用 ==。"""

    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"{name}: expected={expected}, actual={actual}, tolerance={tolerance}"
        )


# =============================================================================
# 2. 本文件使用的两个小型数学函数
# =============================================================================

def fit_affine(
    pixels: list[tuple[float, float]],
    robot_points: list[tuple[float, float]],
) -> np.ndarray:
    """根据像素点和机械臂点，拟合一个 2×3 二维仿射矩阵。"""

    if len(pixels) != len(robot_points) or len(pixels) < 3:
        raise ValueError("二维仿射拟合至少需要三组对应点。")

    # 每个像素点补一个常数 1，得到 [u, v, 1]。
    pixel_matrix = np.column_stack(
        [np.asarray(pixels, dtype=float), np.ones(len(pixels))]
    )
    robot_matrix = np.asarray(robot_points, dtype=float)

    # 最小二乘求解 pixel_matrix @ parameters ≈ robot_matrix。
    result = np.linalg.lstsq(pixel_matrix, robot_matrix, rcond=None)
    parameters = result[0]
    return parameters.T


def directed_long_axis(body_points: np.ndarray, cap_center: np.ndarray) -> np.ndarray:
    """用 PCA 求试管长轴，并让箭头方向指向管盖。"""

    centered = body_points - body_points.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    axis = right_vectors[0]
    body_center = body_points.mean(axis=0)
    if np.dot(cap_center - body_center, axis) < 0:
        axis = -axis
    return axis / np.linalg.norm(axis)


# =============================================================================
# 3. 三个不接硬件的核心检查
# =============================================================================

def test_pixel_to_robot_affine() -> None:
    """拟合已知二维关系，再检查一个没有参与拟合的点。"""

    # 人工构造 x=1+u/100、y=2+v/100 的精确对应关系。
    pixels = [(0, 0), (100, 0), (0, 100), (100, 100)]
    robot_points = [(1, 2), (2, 2), (1, 3), (2, 3)]
    transform = fit_affine(pixels, robot_points)
    x, y = transform @ np.array([50.0, 40.0, 1.0])
    assert_close("affine x", x, 1.5)
    assert_close("affine y", y, 2.4)


def test_tube_geometry() -> None:
    """用合成的细长点云检查试管有向长轴。"""

    center_px = np.array([150.0, 100.0])
    angle_rad = math.radians(30.0)
    expected_axis = np.array([math.cos(angle_rad), math.sin(angle_rad)])

    # 沿长轴生成 81 个点，并沿短轴给每个点增加轻微宽度。
    short_axis = np.array([-expected_axis[1], expected_axis[0]])
    body_point_list = []
    for length in np.linspace(-80.0, 80.0, 81):
        for width in (-8.0, 8.0):
            point = center_px + expected_axis * length + short_axis * width
            body_point_list.append(point)
    body_points = np.array(body_point_list)
    cap_center_px = center_px + expected_axis * 80.0
    axis = directed_long_axis(body_points, cap_center_px)
    # 两个单位向量点积接近 1，说明估计方向与人工构造方向一致。
    direction_dot = float(np.dot(axis, expected_axis))
    if direction_dot < 0.90:
        raise AssertionError(f"tube axis direction is wrong: dot={direction_dot:.3f}")


def test_carm_response_check() -> None:
    """检查 CArm SDK 常见成功字符串，不连接机械臂。"""

    def response_ok(response: object) -> bool:
        """判断 SDK 返回值是否表示命令已被控制器接受。"""

        text = str(response).strip().lower()
        return text in {"task_receive", "task_recieve", "ok", "success"}

    if not response_ok("Task_Receive"):
        raise AssertionError("accepted CArm response was rejected")
    if response_ok("not_ok"):
        raise AssertionError("bad CArm response was accepted")


# =============================================================================
# 4. 主流程
# =============================================================================

def main() -> int:
    """按照本脚本的编号流程依次执行各个步骤。"""

    test_pixel_to_robot_affine()
    test_tube_geometry()
    test_carm_response_check()
    print("Project smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
