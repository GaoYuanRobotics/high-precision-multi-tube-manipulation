#!/usr/bin/env python3
"""用 RealSense 彩色画面实时检查试管分割、图像几何和可选机器人坐标。

本脚本连续读取 RealSense 的 BGR 彩色帧，运行 YOLO 实例分割，并根据管身
mask 的长轴与管盖 mask 的位置叠加二维几何标记：

- ``B``（蓝色）：管身远离管盖的一端，即估计的管底；
- ``C``（红色）：管身靠近管盖的一端，并不是管盖 mask 的中心；
- ``G``（黄色）：``configs/vision.yaml`` 中抓取比例对应的二维像素点；
- 紫色或黄色直线：管身的二维长轴；
- 角度：图像坐标系中从 B 指向 C 的角度；图像 x 轴向右、y 轴向下，
  因此正角度在画面上表现为顺时针。

如果当前帧没有检测到对应管盖，PCA 只能确定一条没有头尾方向的长轴。此时脚本
不会把端点称为 B/C，而会显示 ``E1``、``E2``，把 G 固定在中点，并把角度显示
为模 180 度的轴角；这种结果不能用于确定机械臂抓取朝向。

几何库默认要求管身主轴至少 30 px、PCA 长短轴比至少 2.0；不满足时拒绝该
颜色的几何。状态面板同时显示 ``len`` 和 ``aspect``，便于观察质量余量。

重要边界：

- 这里只打开彩色流，不读取深度、不做彩色与深度对齐；
- 不提供 ``--eye-to-hand`` 时，B/C/G 和角度只在二维图像坐标系中；
- 提供外部工程求得并验证的 ``T_base_from_camera`` 后，脚本会把像素射线与
  固定抓取平面求交，显示 CArm 基座坐标中的 G 和 yaw；
- 本项目不在这里求解手眼标定；本脚本也永远不会连接或控制 CArm；
- 当前尚未实现同色多实例的 body/cap 空间配对；检测到同一颜色多个管身或
  多个盖子时会拒绝输出该颜色几何，因此只适合每种颜色最多一根目标试管。

相关脚本：

- ``07_preview_realtime_seg.py``：只检查实时分割效果；
- ``08_infer_image_geometry.py``：分析一张已保存图片；
- 本脚本：检查 RealSense 彩色流中的实时二维几何。

运行示例：

    conda activate hps
    python scripts/09_preview_realsense_geometry.py \
      --serial REAL_SERIAL

窗口按键：

- ``Q`` 或 ``Esc``：退出；
- ``P`` 或空格：暂停或继续；
- ``S``：保存当前画面；
- ``B``：显示或隐藏检测框；
- ``L``：显示或隐藏模型类别标签；
- ``G``：显示或隐藏二维几何标记。
"""

from __future__ import annotations

# argparse：读取相机、模型、阈值和显示相关的命令行参数。
import argparse
# math：检查有限数值、计算 atan2 yaw，以及弧度/角度转换。
import math
# os：提供原子替换、硬链接和临时文件描述符操作。
import os
# tempfile：保存最后一帧时先写临时图片，避免留下半张文件。
import tempfile
# time：使用高精度单调时钟计算实时帧率。
import time
# Counter：统计当前帧中每个类别出现了多少个实例。
from collections import Counter
# dataclass：定义不可变结果对象；replace：安全复制并更新其中少数字段。
from dataclasses import dataclass, replace
# datetime：按当前时间为按键截图生成不易冲突的文件名。
from datetime import datetime
# Path：规范化、拼接和检查文件路径。
from pathlib import Path
# Any：第三方 YOLO 结果和类别表属于动态类型。
from typing import Any

# OpenCV 负责画窗口、文字、几何点和保存截图。
import cv2
# NumPy 负责图像/掩膜数组及坐标取整。
import numpy as np


# __file__ 指向本脚本；parents[1] 得到项目根目录。
ROOT = Path(__file__).resolve().parents[1]
# =============================================================================
# 本脚本自带的实时几何和外部手眼标定实现
# =============================================================================

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

"""视觉入口共用的严格配置和 JSON 读取工具。

``07``、``08``、``09`` 必须共享同一份四类模型语义。这里不使用带默认值的
``dict.get`` 来“容错”：视觉配置中的拼写错误会改变抓取点或类别配对，因此应在
加载模型、打开相机或执行推理之前立即报错。
"""


import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml



EXPECTED_TUBE_NAMES = ("purple", "yellow")
_EXPECTED_CLASS_PAIRS = {
    "purple": ("p-body", "p-cap"),
    "yellow": ("y-body", "y-cap"),
}
_ROOT_FIELDS = frozenset({"tubes", "grasp"})
_TUBE_FIELDS = frozenset({"body_class", "cap_class", "display_color_bgr"})
_GRASP_FIELDS = frozenset({"fraction_from_bottom"})


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """拒绝所有层级的 YAML 重复键。"""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    """构造 YAML 字典，并拒绝同一层级中的重复键。"""

    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ValueError("YAML 映射键必须是可哈希的简单值。") from exc
        if duplicate:
            raise ValueError(f"YAML 中存在重复键 {key!r}，文件已拒绝。")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """构造 JSON 字典，并拒绝会被静默覆盖的重复键。"""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 中存在重复键 {key!r}，文件已拒绝。")
        result[key] = value
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """确认输入节点确实是字典映射。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} 必须是对象。")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    """确认配置字段既没有缺失，也没有多余拼写。"""

    actual = set(value)
    unknown = sorted(actual - expected, key=repr)
    missing = sorted(expected - actual)
    if unknown:
        raise ValueError(f"{name} 包含未知字段：{unknown}。")
    if missing:
        raise ValueError(f"{name} 缺少必需字段：{missing}。")


def _finite_fraction(value: Any, name: str) -> float:
    """读取 0 到 1 之间的有限比例值。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是 0..1 的有限数值。")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} 必须是 0..1 的有限数值。")
    return result


def _color(value: Any, name: str) -> tuple[int, int, int]:
    """读取三个 0 到 255 的 OpenCV BGR 颜色分量。"""

    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} 必须包含 3 个 0..255 整数。")
    output: list[int] = []
    for index in range(len(value)):
        item = value[index]
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or not 0 <= item <= 255
        ):
            raise ValueError(f"{name}[{index}] 必须是 0..255 的整数。")
        output.append(item)
    return (output[0], output[1], output[2])


@dataclass(frozen=True)
class TubeVisionSpec:
    """一种试管对应的固定类别和 OpenCV 显示颜色。"""

    body_class: str
    cap_class: str
    display_color_bgr: tuple[int, int, int]


@dataclass(frozen=True)
class VisionConfig:
    """经过严格校验的当前四类视觉配置。"""

    tubes: dict[str, TubeVisionSpec]
    grasp_fraction_from_bottom: float

    @property
    def configured_class_order(self) -> tuple[str, ...]:
        """按照配置顺序返回四个模型类别名称。"""

        class_names: list[str] = []
        for tube_name in EXPECTED_TUBE_NAMES:
            tube = self.tubes[tube_name]
            class_names.append(tube.body_class)
            class_names.append(tube.cap_class)
        return tuple(class_names)


def parse_vision_config(
    data: Any,
    *,
    source_name: str = "vision.yaml",
) -> VisionConfig:
    """严格解析内存中的视觉配置，不访问模型、相机或机械臂。"""

    root = _mapping(data, f"{source_name} 顶层")
    _require_exact_fields(root, _ROOT_FIELDS, f"{source_name} 顶层")

    raw_tubes = _mapping(root["tubes"], "tubes")
    actual_tube_order = tuple(raw_tubes)
    if actual_tube_order != EXPECTED_TUBE_NAMES:
        raise ValueError(
            "tubes 必须按 purple、yellow 精确排列且不能增删："
            f"实际={list(actual_tube_order)}，"
            f"期望={list(EXPECTED_TUBE_NAMES)}。"
        )

    tubes: dict[str, TubeVisionSpec] = {}
    for tube_name in EXPECTED_TUBE_NAMES:
        raw_spec = _mapping(raw_tubes[tube_name], f"tubes.{tube_name}")
        _require_exact_fields(
            raw_spec,
            _TUBE_FIELDS,
            f"tubes.{tube_name}",
        )
        expected_body, expected_cap = _EXPECTED_CLASS_PAIRS[tube_name]
        body_class = raw_spec["body_class"]
        cap_class = raw_spec["cap_class"]
        if body_class != expected_body or cap_class != expected_cap:
            raise ValueError(
                f"tubes.{tube_name} 类别必须精确为 "
                f"body_class={expected_body!r}、cap_class={expected_cap!r}，"
                f"实际为 {body_class!r}、{cap_class!r}。"
            )
        tubes[tube_name] = TubeVisionSpec(
            body_class=expected_body,
            cap_class=expected_cap,
            display_color_bgr=_color(
                raw_spec["display_color_bgr"],
                f"tubes.{tube_name}.display_color_bgr",
            ),
        )

    raw_grasp = _mapping(root["grasp"], "grasp")
    _require_exact_fields(raw_grasp, _GRASP_FIELDS, "grasp")
    grasp_fraction = _finite_fraction(
        raw_grasp["fraction_from_bottom"],
        "grasp.fraction_from_bottom",
    )

    config = VisionConfig(
        tubes=tubes,
        grasp_fraction_from_bottom=grasp_fraction,
    )
    # 这里也断言共享模型契约，防止以后只修改一侧的常量。
    if config.configured_class_order != EXPECTED_TUBE_CLASS_ORDER:
        raise ValueError("视觉配置类别顺序与当前四类模型契约不一致。")
    return config


def load_vision_config(path: str | Path) -> VisionConfig:
    """从 YAML 文件加载严格视觉配置并拒绝重复键。"""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"找不到视觉配置：{source}")
    try:
        data = yaml.load(
            source.read_text(encoding="utf-8"),
            Loader=_UniqueKeySafeLoader,
        )
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"无法读取视觉配置 {source}：{exc}") from exc
    return parse_vision_config(data, source_name=str(source))


def load_unique_json_mapping(path: str | Path, *, description: str) -> dict[str, Any]:
    """读取 JSON 顶层对象，并拒绝任意层级的重复键。"""

    source = Path(path).expanduser().resolve()
    try:
        data = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"无法读取{description} {source}：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{description} {source} 顶层必须是 JSON 对象。")
    return data


__all__ = [
    "EXPECTED_TUBE_NAMES",
    "TubeVisionSpec",
    "VisionConfig",
    "load_unique_json_mapping",
    "load_vision_config",
    "parse_vision_config",
]

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

    有管盖掩膜时用它消除长轴 180° 歧义，返回方向始终从无盖端 B 指向管盖端
    C。长度和 PCA 长宽比阈值用于拒绝方向不稳定的小目标或近方形目标。
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

"""导入并使用由其他工程求得的固定相机（eye-to-hand）外参。

本模块只负责读取、校验和使用外部标定结果，不包含任何标定求解，也不会访问
RealSense 或 CArm 硬件。统一采用如下变换方向：

    p_base = T_base_from_camera @ p_camera

其中三维点使用齐次坐标，长度单位统一为米。``camera_frame`` 必须明确指出外参
对应 ``color_optical`` 还是 ``depth_optical``；图像内参也必须来自同一光学流。
"""


import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml


SUPPORTED_CAMERA_FRAMES = frozenset({"color_optical", "depth_optical"})
_ROTATION_ATOL = 1e-4
_HOMOGENEOUS_ATOL = 1e-8

# 桌面固定 D435 的像素射线不应以擦边角度穿过水平工作面。这里使用归一化
# 射线与基座 Z 轴夹角余弦做无量纲检查：0.10 对应射线至少以约 5.7° 入射
# 平面。再把光心到交点限制在 3 m 内，防止虽未达到数学平行、却因外参/平面
# 配置错误而产生数十乃至数十亿米的坐标。两项都是运行时安全边界，并不替代
# 外部标定工程的独立验证。
MIN_RAY_PLANE_NORMAL_COS = 0.10
MAX_RAY_PLANE_DISTANCE_M = 3.0


class _EyeUniqueKeySafeLoader(yaml.SafeLoader):
    """拒绝 YAML 重复键，避免关键矩阵或验证状态被静默覆盖。"""


def _eye_construct_unique_mapping(
    loader: _EyeUniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    """构造手眼标定 YAML 字典，并拒绝重复键。"""

    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ValueError("YAML 映射键必须是可哈希的简单值。") from exc
        if duplicate:
            raise ValueError(f"YAML 中存在重复键 {key!r}，标定文件已拒绝。")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_EyeUniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _eye_construct_unique_mapping,
)


def _eye_unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """供 json.loads 使用的重复键拒绝钩子。"""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 中存在重复键 {key!r}，标定文件已拒绝。")
        result[key] = value
    return result


def _eye_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """确认配置节点为映射，并给出面向用户的字段名。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} 必须是 YAML/JSON 对象。")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    """读取不允许为空的字符串字段。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串。")
    return value.strip()


def _finite_float(value: Any, name: str) -> float:
    """读取有限浮点数，拒绝 NaN 和无穷大。"""

    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是数值，不能是布尔值。")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数值。") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} 必须是有限数值，不能是 NaN 或无穷大。")
    return result


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    """读取整数，避免把 1280.5 之类的数静默截断。"""

    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是整数。")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数。") from exc
    if result != value:
        raise ValueError(f"{name} 必须是整数。")
    minimum = 1
    if allow_zero:
        minimum = 0
    if result < minimum:
        comparator = "正"
        if allow_zero:
            comparator = "非负"
        raise ValueError(f"{name} 必须是{comparator}整数。")
    return result


@dataclass(frozen=True)
class CameraIntrinsics:
    """与外参对应的针孔相机内参，坐标单位为像素。

    支持已整流图像的 ``none`` 模型，以及 OpenCV 可迭代反投影的
    ``brown_conrady``。其他 RealSense 畸变模型不会被静默近似。
    """

    fx: float
    fy: float
    ppx: float
    ppy: float
    distortion_model: str
    coeffs: tuple[float, ...]

    def __post_init__(self) -> None:
        """对象创建后立即校验字段形状、单位和有限值。"""

        values = {
            "intrinsics.fx": self.fx,
            "intrinsics.fy": self.fy,
            "intrinsics.ppx": self.ppx,
            "intrinsics.ppy": self.ppy,
        }
        normalized: dict[str, float] = {}
        for name, value in values.items():
            normalized[name] = _finite_float(value, name)
        if normalized["intrinsics.fx"] <= 0 or normalized["intrinsics.fy"] <= 0:
            raise ValueError("intrinsics.fx 和 intrinsics.fy 必须大于 0。")
        distortion_model = _nonempty_string(
            self.distortion_model, "intrinsics.distortion_model"
        ).lower()
        aliases = {
            "none": "none",
            "no_distortion": "none",
            "brown_conrady": "brown_conrady",
            "opencv_brown_conrady": "brown_conrady",
            "inverse_brown_conrady": "zero_distortion_alias",
            "modified_brown_conrady": "zero_distortion_alias",
        }
        if distortion_model not in aliases:
            raise ValueError(
                "当前像素射线实现只支持 distortion_model=none 或 "
                "brown_conrady；"
                f"intrinsics.distortion_model={distortion_model!r}。"
                "不受支持的模型不会被静默近似；请在外部先整流并导出新内参。"
            )
        distortion_model = aliases[distortion_model]
        if isinstance(self.coeffs, (str, bytes)) or not isinstance(
            self.coeffs, Sequence
        ):
            raise ValueError("intrinsics.coeffs 必须是数值数组。")
        coefficient_values: list[float] = []
        for index in range(len(self.coeffs)):
            value = self.coeffs[index]
            coefficient_values.append(
                _finite_float(value, f"intrinsics.coeffs[{index}]")
            )
        coefficients = tuple(coefficient_values)
        if distortion_model == "zero_distortion_alias":
            has_nonzero_coefficient = False
            for value in coefficients:
                if abs(value) > 1e-12:
                    has_nonzero_coefficient = True
                    break
            if has_nonzero_coefficient:
                raise ValueError(
                    "inverse/modified Brown-Conrady 只有在全部畸变系数严格为 0 "
                    "时才能等价归一化为 none；非零系数不会被静默忽略。"
                )
            distortion_model = "none"
        if distortion_model == "none":
            has_nonzero_coefficient = False
            for value in coefficients:
                if abs(value) > 1e-12:
                    has_nonzero_coefficient = True
                    break
            if has_nonzero_coefficient:
                raise ValueError(
                    "distortion_model=none 时 intrinsics.coeffs 必须为空或"
                    "全部为 0；当前实现不会静默忽略非零畸变系数。"
                )
        elif len(coefficients) not in {4, 5, 8, 12, 14}:
            raise ValueError(
                "brown_conrady 的 intrinsics.coeffs 长度必须是 OpenCV "
                "支持的 4、5、8、12 或 14。"
            )
        object.__setattr__(self, "fx", normalized["intrinsics.fx"])
        object.__setattr__(self, "fy", normalized["intrinsics.fy"])
        object.__setattr__(self, "ppx", normalized["intrinsics.ppx"])
        object.__setattr__(self, "ppy", normalized["intrinsics.ppy"])
        object.__setattr__(self, "distortion_model", distortion_model)
        object.__setattr__(self, "coeffs", coefficients)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CameraIntrinsics":
        """从 YAML/JSON 内参节点构造对象。"""

        required = (
            "fx",
            "fy",
            "ppx",
            "ppy",
            "distortion_model",
            "coeffs",
        )
        missing: list[str] = []
        for key in required:
            if key not in data:
                missing.append(key)
        if missing:
            raise ValueError(f"intrinsics 缺少字段：{', '.join(missing)}")
        coefficients_value = data["coeffs"]
        if isinstance(coefficients_value, Sequence) and not isinstance(
            coefficients_value, (str, bytes)
        ):
            coefficients_value = tuple(coefficients_value)
        return cls(
            fx=data["fx"],
            fy=data["fy"],
            ppx=data["ppx"],
            ppy=data["ppy"],
            distortion_model=data["distortion_model"],
            coeffs=coefficients_value,
        )

    def as_dict(self) -> dict[str, float]:
        """返回适合写入 JSON 的内参字典。"""

        return {
            "fx": self.fx,
            "fy": self.fy,
            "ppx": self.ppx,
            "ppy": self.ppy,
            "distortion_model": self.distortion_model,
            "coeffs": list(self.coeffs),
        }


@dataclass(frozen=True)
class ValidationMetadata:
    """外部工程对标定结果所做的独立验证记录。

    ``status=validated`` 时，日期、验证点数和米制误差均为必填项；模板或尚未验证
    的结果可以使用 ``status=unvalidated``，但运动代码应拒绝这种状态。
    """

    status: str
    method: str
    validated_at: str | None
    point_count: int
    mean_error_m: float | None
    max_error_m: float | None
    notes: str = ""

    def __post_init__(self) -> None:
        """对象创建后立即校验字段形状、单位和有限值。"""

        status = _nonempty_string(self.status, "validation.status").lower()
        if status not in {"validated", "unvalidated"}:
            raise ValueError(
                "validation.status 只能是 'validated' 或 'unvalidated'。"
            )
        method = _nonempty_string(self.method, "validation.method")
        point_count = _positive_int(
            self.point_count, "validation.point_count", allow_zero=True
        )
        validated_at = self.validated_at
        if validated_at is not None:
            validated_at = _nonempty_string(
                validated_at, "validation.validated_at"
            )
            parse_text = validated_at
            if validated_at.endswith(("Z", "z")):
                parse_text = f"{validated_at[:-1]}+00:00"
            try:
                datetime.fromisoformat(parse_text)
            except ValueError as exc:
                raise ValueError(
                    "validation.validated_at 必须是有效 ISO 8601 时间。"
                ) from exc

        mean_error = None
        if self.mean_error_m is not None:
            mean_error = _finite_float(
                self.mean_error_m, "validation.mean_error_m"
            )
        max_error = None
        if self.max_error_m is not None:
            max_error = _finite_float(self.max_error_m, "validation.max_error_m")
        for name, value in (
            ("validation.mean_error_m", mean_error),
            ("validation.max_error_m", max_error),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} 不能为负数。")

        if status == "validated":
            if validated_at is None:
                raise ValueError(
                    "validation.status=validated 时必须填写 validated_at。"
                )
            if point_count < 4:
                raise ValueError(
                    "validation.status=validated 时 point_count 必须至少为 4；"
                    "一个或少量检查点不足以覆盖工作区域。"
                )
            if mean_error is None or max_error is None:
                raise ValueError(
                    "validation.status=validated 时必须填写 mean_error_m 和 "
                    "max_error_m。"
                )
            if max_error + 1e-15 < mean_error:
                raise ValueError(
                    "validation.max_error_m 不能小于 mean_error_m。"
                )

        object.__setattr__(self, "status", status)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "validated_at", validated_at)
        object.__setattr__(self, "point_count", point_count)
        object.__setattr__(self, "mean_error_m", mean_error)
        object.__setattr__(self, "max_error_m", max_error)
        object.__setattr__(self, "notes", str(self.notes or ""))

    @property
    def is_validated(self) -> bool:
        """结果是否已被外部流程标记为验证通过。"""

        return self.status == "validated"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ValidationMetadata":
        """从配置中的 ``validation`` 节点构造验证记录。"""

        allowed = {
            "status",
            "method",
            "validated_at",
            "point_count",
            "mean_error_m",
            "max_error_m",
            "notes",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"validation 包含未知字段：{sorted(unknown)}")
        required = ("status", "method", "point_count")
        missing: list[str] = []
        for key in required:
            if key not in data:
                missing.append(key)
        if missing:
            raise ValueError(f"validation 缺少字段：{', '.join(missing)}")
        return cls(
            status=data["status"],
            method=data["method"],
            validated_at=data.get("validated_at"),
            point_count=data["point_count"],
            mean_error_m=data.get("mean_error_m"),
            max_error_m=data.get("max_error_m"),
            notes=data.get("notes", ""),
        )

    def as_dict(self) -> dict[str, Any]:
        """返回可序列化的验证元数据。"""

        return {
            "status": self.status,
            "method": self.method,
            "validated_at": self.validated_at,
            "point_count": self.point_count,
            "mean_error_m": self.mean_error_m,
            "max_error_m": self.max_error_m,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ExternalEyeToHandCalibration:
    """经过结构校验的固定相机外参、内参与工作平面信息。"""

    T_base_from_camera: np.ndarray
    camera_frame: str
    camera_serial: str
    width: int
    height: int
    intrinsics: CameraIntrinsics
    image_is_rectified: bool
    plane_z_m: float
    validation: ValidationMetadata
    schema_version: int = 1

    def __post_init__(self) -> None:
        """对象创建后立即校验字段形状、单位和有限值。"""

        matrix = np.asarray(self.T_base_from_camera, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(
                "T_base_from_camera 必须是 4×4 矩阵，"
                f"当前形状为 {matrix.shape}。"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                "T_base_from_camera 必须全部为有限数值，不能包含 NaN 或无穷大。"
            )
        if not np.allclose(
            matrix[3],
            np.array([0.0, 0.0, 0.0, 1.0]),
            atol=_HOMOGENEOUS_ATOL,
            rtol=0.0,
        ):
            raise ValueError(
                "T_base_from_camera 的齐次底行必须为 [0, 0, 0, 1]。"
            )

        rotation = matrix[:3, :3]
        orthogonal_error = float(
            np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro")
        )
        determinant = float(np.linalg.det(rotation))
        if orthogonal_error > _ROTATION_ATOL:
            raise ValueError(
                "T_base_from_camera 的旋转矩阵不正交："
                f"||R^T R-I||={orthogonal_error:.3e}，"
                f"允许上限为 {_ROTATION_ATOL:.1e}。"
            )
        if not np.isclose(
            determinant, 1.0, atol=_ROTATION_ATOL, rtol=0.0
        ):
            raise ValueError(
                "T_base_from_camera 的旋转矩阵必须是右手系且 det(R)=1，"
                f"当前 det(R)={determinant:.8f}。"
            )

        camera_frame = _nonempty_string(
            self.camera_frame, "camera.frame"
        ).lower()
        if camera_frame not in SUPPORTED_CAMERA_FRAMES:
            choices = ", ".join(sorted(SUPPORTED_CAMERA_FRAMES))
            raise ValueError(f"camera.frame 必须是以下之一：{choices}。")
        camera_serial = _nonempty_string(self.camera_serial, "camera.serial")
        width = _positive_int(self.width, "camera.width")
        height = _positive_int(self.height, "camera.height")
        if not isinstance(self.image_is_rectified, bool):
            raise ValueError(
                "camera.image_is_rectified 必须明确填写 true 或 false。"
            )
        if self.image_is_rectified and self.intrinsics.distortion_model != "none":
            raise ValueError(
                "camera.image_is_rectified=true 时必须使用整流后内参，并设置 "
                "distortion_model: none；不能继续携带原始畸变模型。"
            )
        if (
            not self.image_is_rectified
            and self.intrinsics.distortion_model != "brown_conrady"
        ):
            raise ValueError(
                "camera.image_is_rectified=false 时当前只支持 "
                "distortion_model: brown_conrady。"
            )
        plane_z = _finite_float(self.plane_z_m, "plane_z_m")
        schema_version = _positive_int(
            self.schema_version, "schema_version"
        )
        if schema_version != 1:
            raise ValueError(
                f"不支持 schema_version={schema_version}，当前只支持版本 1。"
            )
        if not isinstance(self.intrinsics, CameraIntrinsics):
            raise ValueError("intrinsics 必须是 CameraIntrinsics。")
        if not isinstance(self.validation, ValidationMetadata):
            raise ValueError("validation 必须是 ValidationMetadata。")
        if not (0.0 <= self.intrinsics.ppx < width):
            raise ValueError(
                f"intrinsics.ppx={self.intrinsics.ppx} 超出图像宽度 [0, {width})。"
            )
        if not (0.0 <= self.intrinsics.ppy < height):
            raise ValueError(
                f"intrinsics.ppy={self.intrinsics.ppy} 超出图像高度 [0, {height})。"
            )

        # 复制并设为只读，防止加载后被调用方无意修改而绕过上述校验。
        matrix = matrix.copy()
        matrix.setflags(write=False)
        object.__setattr__(self, "T_base_from_camera", matrix)
        object.__setattr__(self, "camera_frame", camera_frame)
        object.__setattr__(self, "camera_serial", camera_serial)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(
            self, "image_is_rectified", self.image_is_rectified
        )
        object.__setattr__(self, "plane_z_m", plane_z)
        object.__setattr__(self, "schema_version", schema_version)

    @property
    def rotation_base_from_camera(self) -> np.ndarray:
        """相机光学坐标轴到机械臂基座坐标轴的 3×3 旋转。"""

        return self.T_base_from_camera[:3, :3]

    @property
    def camera_origin_base_m(self) -> np.ndarray:
        """相机光心在机械臂基座坐标系中的位置，单位为米。"""

        return self.T_base_from_camera[:3, 3]

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any]
    ) -> "ExternalEyeToHandCalibration":
        """从已解析的 YAML/JSON 对象创建标定对象。"""

        root = _eye_mapping(data, "标定文件根节点")
        allowed_root = {
            "schema_version",
            "type",
            "units",
            "transform_convention",
            "T_base_from_camera",
            "camera",
            "plane_z_m",
            "validation",
        }
        unknown_root = set(root) - allowed_root
        if unknown_root:
            raise ValueError(
                f"标定文件根节点包含未知字段：{sorted(unknown_root)}"
            )
        calibration_type = _nonempty_string(root.get("type"), "type")
        if calibration_type != "external_eye_to_hand":
            raise ValueError(
                "type 必须是 'external_eye_to_hand'，避免误读其他标定文件。"
            )
        convention = _nonempty_string(
            root.get("transform_convention"), "transform_convention"
        )
        if convention != "T_base_from_camera":
            raise ValueError(
                "transform_convention 必须是 'T_base_from_camera'；"
                "本项目统一使用 p_base = T_base_from_camera @ p_camera，"
                "不会自动猜测或反转矩阵。"
            )
        units = _nonempty_string(root.get("units"), "units").lower()
        if units not in {"m", "meter", "metre"}:
            raise ValueError(
                "units 必须为 'm'（米）；毫米结果必须先除以 1000 再导入。"
            )
        if "T_base_from_camera" not in root:
            raise ValueError("标定文件缺少 T_base_from_camera。")

        camera = _eye_mapping(root.get("camera"), "camera")
        allowed_camera = {
            "frame",
            "serial",
            "width",
            "height",
            "image_is_rectified",
            "intrinsics",
        }
        unknown_camera = set(camera) - allowed_camera
        if unknown_camera:
            raise ValueError(
                f"camera 包含未知字段：{sorted(unknown_camera)}"
            )
        raw_intrinsics = _eye_mapping(
            camera.get("intrinsics"),
            "camera.intrinsics",
        )
        allowed_intrinsics = {
            "fx",
            "fy",
            "ppx",
            "ppy",
            "distortion_model",
            "coeffs",
        }
        unknown_intrinsics = set(raw_intrinsics) - allowed_intrinsics
        if unknown_intrinsics:
            raise ValueError(
                "camera.intrinsics 包含未知字段："
                f"{sorted(unknown_intrinsics)}"
            )
        intrinsics = CameraIntrinsics.from_mapping(
            raw_intrinsics
        )
        validation = ValidationMetadata.from_mapping(
            _eye_mapping(root.get("validation"), "validation")
        )
        if "plane_z_m" not in root:
            raise ValueError(
                "标定文件缺少 plane_z_m；它是像素射线求交的基座坐标 Z 平面。"
            )
        try:
            transform = np.asarray(
                root["T_base_from_camera"], dtype=np.float64
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "T_base_from_camera 必须是由数值组成的规则 4×4 数组。"
            ) from exc

        return cls(
            T_base_from_camera=transform,
            camera_frame=camera.get("frame"),
            camera_serial=camera.get("serial"),
            width=camera.get("width"),
            height=camera.get("height"),
            intrinsics=intrinsics,
            image_is_rectified=camera.get("image_is_rectified"),
            plane_z_m=root["plane_z_m"],
            validation=validation,
            schema_version=root.get("schema_version", 1),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExternalEyeToHandCalibration":
        """从 ``.yaml/.yml`` 或 ``.json`` 文件导入并立即严格校验。"""

        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"找不到外部眼在手外标定文件：{source}")
        suffix = source.suffix.lower()
        try:
            text = source.read_text(encoding="utf-8")
            if suffix == ".json":
                data = json.loads(text, object_pairs_hook=_eye_unique_json_object)
            elif suffix in {".yaml", ".yml"}:
                data = yaml.load(text, Loader=_EyeUniqueKeySafeLoader)
            else:
                raise ValueError(
                    f"不支持标定文件扩展名 '{source.suffix}'，"
                    "只支持 .yaml、.yml 和 .json。"
                )
        except (json.JSONDecodeError, yaml.YAMLError) as exc:
            raise ValueError(f"无法解析标定文件 {source}：{exc}") from exc
        return cls.from_mapping(_eye_mapping(data, "标定文件根节点"))

    def validate_stream(
        self,
        serial: str,
        width: int,
        height: int,
        camera_frame: str | None = None,
    ) -> None:
        """确认实时流与标定时的相机、分辨率和光学坐标系完全一致。

        不匹配时抛出 ``ValueError``，调用方不能继续输出机器人目标坐标。
        """

        actual_serial = _nonempty_string(serial, "实时相机 serial")
        actual_width = _positive_int(width, "实时图像 width")
        actual_height = _positive_int(height, "实时图像 height")
        mismatches: list[str] = []
        if actual_serial != self.camera_serial:
            mismatches.append(
                f"serial 当前为 {actual_serial!r}，标定值为 "
                f"{self.camera_serial!r}"
            )
        if actual_width != self.width or actual_height != self.height:
            mismatches.append(
                f"分辨率当前为 {actual_width}×{actual_height}，"
                f"标定值为 {self.width}×{self.height}"
            )
        if camera_frame is not None:
            actual_frame = _nonempty_string(
                camera_frame, "实时 camera_frame"
            ).lower()
            if actual_frame != self.camera_frame:
                mismatches.append(
                    f"camera_frame 当前为 {actual_frame!r}，"
                    f"标定值为 {self.camera_frame!r}"
                )
        if mismatches:
            raise ValueError("实时相机流与外参不匹配：" + "；".join(mismatches) + "。")

    def validate_intrinsics(
        self,
        *,
        fx: float,
        fy: float,
        ppx: float,
        ppy: float,
        distortion_model: str,
        coeffs: Sequence[float],
        pixel_atol: float = 1e-4,
        coeff_atol: float = 1e-7,
    ) -> None:
        """确认当前流内参与外部标定文件一致。

        RealSense 常把模型打印为 ``distortion.brown_conrady``，这里仅做名称
        归一化，不会把 inverse/modified 模型近似成普通 Brown-Conrady。
        """

        model_text = _nonempty_string(
            distortion_model,
            "实时 intrinsics.distortion_model",
        ).lower()
        model_text = model_text.replace("distortion.", "").replace("-", "_")
        model_aliases = {
            "none": "none",
            "no_distortion": "none",
            "brown_conrady": "brown_conrady",
        }
        try:
            actual_coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError("实时 intrinsics.coeffs 必须是数值数组。") from exc
        if not np.all(np.isfinite(actual_coeffs)):
            raise ValueError("实时 intrinsics.coeffs 不能包含 NaN 或无穷大。")

        actual_model = model_aliases.get(model_text)
        if actual_model is None and model_text in {
            "inverse_brown_conrady",
            "modified_brown_conrady",
        }:
            if np.any(np.abs(actual_coeffs) > coeff_atol):
                raise ValueError(
                    "实时 inverse/modified Brown-Conrady 含非零系数，"
                    "不能安全地当作 none；请在外部整流。"
                )
            actual_model = "none"
        if actual_model is None:
            raise ValueError(
                "实时相机畸变模型不受支持或与外参格式不兼容："
                f"{distortion_model!r}。不会自动近似为 Brown-Conrady。"
            )

        actual_values = np.asarray(
            [
                _finite_float(fx, "实时 intrinsics.fx"),
                _finite_float(fy, "实时 intrinsics.fy"),
                _finite_float(ppx, "实时 intrinsics.ppx"),
                _finite_float(ppy, "实时 intrinsics.ppy"),
            ],
            dtype=np.float64,
        )
        expected_values = np.asarray(
            [
                self.intrinsics.fx,
                self.intrinsics.fy,
                self.intrinsics.ppx,
                self.intrinsics.ppy,
            ],
            dtype=np.float64,
        )
        if not np.allclose(
            actual_values,
            expected_values,
            atol=float(pixel_atol),
            rtol=0.0,
        ):
            raise ValueError(
                "当前相机 fx/fy/ppx/ppy 与 eye-to-hand 标定文件不一致："
                f"当前={actual_values.tolist()}，"
                f"标定={expected_values.tolist()}。"
            )
        if actual_model != self.intrinsics.distortion_model:
            raise ValueError(
                "当前相机 distortion_model 与 eye-to-hand 标定文件不一致："
                f"当前={actual_model!r}，"
                f"标定={self.intrinsics.distortion_model!r}。"
            )

        expected_coeffs = np.asarray(self.intrinsics.coeffs, dtype=np.float64)
        # RealSense 的 none 模型也可能固定返回 5 个零；只比较有效非零语义。
        if actual_model == "none":
            if np.any(np.abs(actual_coeffs) > coeff_atol):
                raise ValueError("实时 none 畸变模型却带有非零畸变系数。")
        elif (
            actual_coeffs.shape != expected_coeffs.shape
            or not np.allclose(
                actual_coeffs,
                expected_coeffs,
                atol=float(coeff_atol),
                rtol=0.0,
            )
        ):
            raise ValueError(
                "当前相机畸变系数与 eye-to-hand 标定文件不一致。"
            )

    def camera_points_to_base(
        self, points_camera_m: Sequence[Sequence[float]] | Sequence[float]
    ) -> np.ndarray:
        """把一个或一批相机光学坐标三维点变换到基座坐标，单位均为米。"""

        try:
            points = np.asarray(points_camera_m, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("相机三维点必须由数值组成。") from exc
        single = points.ndim == 1
        if single:
            if points.shape != (3,):
                raise ValueError("单个相机三维点必须包含 x,y,z 三个数值。")
            points = points.reshape(1, 3)
        elif points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("相机三维点必须是形状为 (3,) 或 (N,3) 的数组。")
        if not np.all(np.isfinite(points)):
            raise ValueError("相机三维点不能包含 NaN 或无穷大。")

        homogeneous = np.column_stack(
            [points, np.ones(len(points), dtype=np.float64)]
        )
        transformed = homogeneous @ self.T_base_from_camera.T
        result = transformed[:, :3]
        if single:
            return result[0]
        return result

    def _checked_uv(self, uv: Sequence[float]) -> tuple[float, float]:
        """校验一个像素坐标并返回浮点值。"""

        try:
            if len(uv) != 2:
                raise ValueError
            u = _finite_float(uv[0], "像素 u")
            v = _finite_float(uv[1], "像素 v")
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc):
                raise
            raise ValueError("像素坐标必须恰好包含 (u, v) 两个数值。") from exc
        if not (0.0 <= u < self.width and 0.0 <= v < self.height):
            raise ValueError(
                f"像素 ({u:.3f}, {v:.3f}) 超出标定图像范围 "
                f"[0,{self.width})×[0,{self.height})。"
            )
        return u, v

    def pixel_to_base_plane(
        self,
        uv: Sequence[float],
        plane_z_m: float | None = None,
    ) -> tuple[float, float, float]:
        """将像素射线与基座 ``z=plane_z_m`` 平面求交。

        这里使用与 ``camera_frame`` 对应的针孔内参。若平面位于相机射线后方，或
        射线与平面近似平行，会明确报错而不是返回危险坐标。针对本项目桌面
        D435，归一化射线在基座 Z 方向的绝对分量必须至少为
        ``MIN_RAY_PLANE_NORMAL_COS=0.10``，且光心到交点不得超过
        ``MAX_RAY_PLANE_DISTANCE_M=3.0 m``。
        """

        u, v = self._checked_uv(uv)
        plane_z = self.plane_z_m
        if plane_z_m is not None:
            plane_z = _finite_float(plane_z_m, "plane_z_m")

        if self.intrinsics.distortion_model == "brown_conrady":
            camera_matrix = np.array(
                [
                    [self.intrinsics.fx, 0.0, self.intrinsics.ppx],
                    [0.0, self.intrinsics.fy, self.intrinsics.ppy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            # OpenCV 对 Brown-Conrady 做迭代反投影，输出无畸变归一化坐标。
            normalized = cv2.undistortPoints(
                np.array([[[u, v]]], dtype=np.float64),
                camera_matrix,
                np.asarray(self.intrinsics.coeffs, dtype=np.float64),
            )[0, 0]
            if not np.all(np.isfinite(normalized)):
                raise ValueError(
                    "Brown-Conrady 像素反投影未得到有限结果；请检查内参和系数。"
                )
            ray_x, ray_y = float(normalized[0]), float(normalized[1])
        else:
            ray_x = (u - self.intrinsics.ppx) / self.intrinsics.fx
            ray_y = (v - self.intrinsics.ppy) / self.intrinsics.fy
        ray_camera = np.array([ray_x, ray_y, 1.0], dtype=np.float64)
        ray_base = self.rotation_base_from_camera @ ray_camera
        origin = self.camera_origin_base_m
        ray_norm = float(np.linalg.norm(ray_base))
        if not math.isfinite(ray_norm) or ray_norm <= 1e-12:
            raise ValueError("像素反投影得到无效的零长度射线。")
        ray_base_unit = ray_base / ray_norm
        normal_cos = abs(float(ray_base_unit[2]))
        if normal_cos < MIN_RAY_PLANE_NORMAL_COS:
            raise ValueError(
                "该像素射线与目标基座 Z 平面近乎平行（入射角过小），"
                "无法安全计算交点："
                f"|方向·Z轴|={normal_cos:.6f}，要求至少为 "
                f"{MIN_RAY_PLANE_NORMAL_COS:.2f}。"
            )
        distance = (plane_z - float(origin[2])) / float(ray_base_unit[2])
        if distance <= 0.0:
            raise ValueError(
                "目标平面位于该像素射线的相机后方；请检查 "
                "T_base_from_camera 的方向、camera.frame 和 plane_z_m。"
            )
        if not math.isfinite(distance) or distance > MAX_RAY_PLANE_DISTANCE_M:
            raise ValueError(
                "光心到工作平面交点的距离超出本项目桌面 D435 安全上限："
                f"{distance:.6f} m > {MAX_RAY_PLANE_DISTANCE_M:.1f} m；"
                "请检查外参方向、相机坐标系和 plane_z_m。"
            )
        point = origin + distance * ray_base_unit
        # 消除浮点累积，让返回值的 Z 与请求平面严格一致。
        point[2] = plane_z
        return float(point[0]), float(point[1]), float(point[2])

    def pixels_to_base_plane(
        self,
        uv_points: Sequence[Sequence[float]],
        plane_z_m: float | None = None,
    ) -> np.ndarray:
        """批量把 ``N×2`` 像素坐标转换成 ``N×3`` 基座平面坐标。"""

        try:
            points = np.asarray(uv_points, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("批量像素坐标必须由数值组成。") from exc
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("批量像素坐标必须是形状为 (N,2) 的数组。")
        if len(points) == 0:
            return np.empty((0, 3), dtype=np.float64)
        converted_points: list[tuple[float, float, float]] = []
        for point in points:
            converted_point = self.pixel_to_base_plane(
                point, plane_z_m=plane_z_m
            )
            converted_points.append(converted_point)
        return np.asarray(converted_points, dtype=np.float64)

    def bcg_to_base(
        self,
        b_uv: Sequence[float],
        c_uv: Sequence[float],
        g_uv: Sequence[float],
        plane_z_m: float | None = None,
    ) -> dict[str, tuple[float, float, float]]:
        """一次转换几何模块的 B、C、G 三个像素点。"""

        converted = self.pixels_to_base_plane(
            [b_uv, c_uv, g_uv], plane_z_m=plane_z_m
        )
        output: dict[str, tuple[float, float, float]] = {}
        names = ("B", "C", "G")
        for index in range(len(names)):
            point = converted[index]
            point_values: list[float] = []
            for value in point:
                point_values.append(float(value))
            output[names[index]] = tuple(point_values)
        return output

    def as_dict(self) -> dict[str, Any]:
        """返回当前规范格式，便于检查或由上层程序记录快照。"""

        return {
            "schema_version": self.schema_version,
            "type": "external_eye_to_hand",
            "units": "m",
            "transform_convention": "T_base_from_camera",
            "T_base_from_camera": self.T_base_from_camera.tolist(),
            "camera": {
                "frame": self.camera_frame,
                "serial": self.camera_serial,
                "width": self.width,
                "height": self.height,
                "image_is_rectified": self.image_is_rectified,
                "intrinsics": self.intrinsics.as_dict(),
            },
            "plane_z_m": self.plane_z_m,
            "validation": self.validation.as_dict(),
        }
@dataclass(frozen=True)
class TubeSpec:
    """一种试管的管身类别、管盖类别和显示颜色。

    ``frozen=True`` 让配置在实时循环中保持不可变，防止某一帧意外改掉后续帧
    的类别语义。使用字段名也比传递三元素 tuple 更容易让初学者理解。
    """

    # 例如 p-body 或 y-body。
    body_class: str
    # 例如 p-cap 或 y-cap，用来消除长轴 180° 歧义。
    cap_class: str
    # OpenCV 颜色顺序是蓝、绿、红（BGR），不是常见的 RGB。
    display_color_bgr: tuple[int, int, int]


@dataclass(frozen=True)
class MaskDetection:
    """某个类别中置信度最高的分割实例。"""

    # 与原始彩色帧同高同宽的二维布尔数组，True 表示前景。
    mask: np.ndarray
    # YOLO 对该实例的置信度，通常位于 0..1。
    confidence: float


@dataclass(frozen=True)
class TubeFrameGeometry:
    """一帧中某种试管的检测和二维几何结果。

    多数字段允许为 ``None``，因为实时画面可能漏检管身、漏检管盖、几何质量
    不足，或没有提供 eye-to-hand。``error`` 属于图像几何错误，
    ``robot_error`` 则属于像素到基座平面的转换错误，两者分开记录。
    """

    # vision.yaml 中的逻辑名称，例如 purple、yellow。
    tube_name: str
    # 与该颜色对应的 body/cap 类别和显示颜色。
    spec: TubeSpec
    # 几何计算成功时保存 TubePose2D，失败/未检测时为 None。
    pose: TubePose2D | None
    # 分别保存 body 和 cap 的置信度；缺失的目标用 None。
    body_confidence: float | None
    cap_confidence: float | None
    # 图像检测或二维几何失败原因；成功且有管盖时通常为 None。
    error: str | None = None
    # 提供外部标定且求交成功时填写机器人坐标；失败时坐标保持 None，并把
    # 该目标的失败原因写入 robot_error。
    robot_grasp_xyz_m: tuple[float, float, float] | None = None
    robot_bottom_xyz_m: tuple[float, float, float] | None = None
    robot_cap_xyz_m: tuple[float, float, float] | None = None
    robot_yaw_rad: float | None = None
    robot_error: str | None = None

    @property
    def direction_resolved(self) -> bool:
        """是否由管盖检测稳定确定了管底到管盖的方向。"""

        # 有 pose 只能说明长轴存在；还必须有管盖才知道哪端是 C。
        return self.pose is not None and self.cap_confidence is not None


class RealSenseColorSource:
    """只打开 RealSense 彩色流，不创建深度流或对齐对象。

    把相机生命周期封装成类，可以让 ``main`` 只调用 ``read`` 和 ``close``；
    初始化任何一步失败时，本类也会负责停止已经启动的 pipeline。
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        serial: str | None,
    ) -> None:
        """选择唯一 RealSense，并启动指定规格的 BGR 彩色流。

        ``width``、``height``、``fps`` 是请求规格；启动后仍从实际 profile
        回读设备序列号、流规格和内参。初始化途中任何一步失败，已经启动的
        pipeline 都会在异常分支中停止。这里不会创建深度流或对齐对象。
        """

        # 延迟导入：只有真正创建 RealSense 帧源时才要求安装 SDK。
        import pyrealsense2 as rs

        # pipeline 管理 RealSense 数据流；先记为未启动，便于异常清理。
        self._pipeline = rs.pipeline()
        self._started = False
        # 枚举真实设备。若连接多台但未传 serial，此函数会拒绝随机选择。
        selected_serial = select_realsense_device_serial(rs, serial)
        # config 用来声明“打开哪台设备、哪种流、什么分辨率和帧率”。
        config = rs.config()
        # 即使只有一台设备也固定枚举到的序列号，禁止 SDK 随机选择。
        config.enable_device(selected_serial)
        # rs.format.bgr8 可直接交给 OpenCV 和 Ultralytics，不需要交换通道。
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        try:
            # start 真正占用设备，并返回相机实际接受的 profile。
            profile = self._pipeline.start(config)
            self._started = True

            # 不信任仅靠请求值；从实际 profile 再读取设备身份。
            device = profile.get_device()
            device_name = device.get_info(rs.camera_info.name)
            device_serial = device.get_info(rs.camera_info.serial_number)
            if str(device_serial) != selected_serial:
                raise RuntimeError(
                    "RealSense 实际启动设备与预选序列号不一致："
                    f"selected={selected_serial!r}, actual={device_serial!r}"
                )
            # get_stream 回读启动后的实际彩色流规格，而不把请求值当作验证事实。
            color_profile = profile.get_stream(
                rs.stream.color
            ).as_video_stream_profile()
            # 内参用于把像素 (u,v) 反投影为相机坐标系射线。
            color_intrinsics = color_profile.get_intrinsics()
            self.serial = str(device_serial)
            self.width = int(color_profile.width())
            self.height = int(color_profile.height())
            self.fps = float(color_profile.fps())
            # 转成普通 Python 数值，方便严格标定校验函数读取。
            coefficient_values: list[float] = []
            for value in color_intrinsics.coeffs:
                coefficient_values.append(float(value))
            self.intrinsics = {
                "fx": float(color_intrinsics.fx),
                "fy": float(color_intrinsics.fy),
                "ppx": float(color_intrinsics.ppx),
                "ppy": float(color_intrinsics.ppy),
                "distortion_model": str(color_intrinsics.model),
                "coeffs": coefficient_values,
            }
            self.description = (
                f"{device_name} serial={device_serial} "
                f"color={self.width}x{self.height}@{self.fps:g}"
            )
        except Exception:
            # 初始化中途任何异常都释放相机，再把原异常继续抛给上层。
            if self._started:
                self._pipeline.stop()
                self._started = False
            raise

    def read(self) -> np.ndarray | None:
        """等待下一张彩色帧，并以 OpenCV BGR 数组返回。"""

        # 最多等待 5 秒，避免设备异常时永久卡住。
        frames = self._pipeline.wait_for_frames(timeout_ms=5000)
        color_frame = frames.get_color_frame()
        # SDK 可能返回一个不含彩色帧的 frameset，此帧直接跳过。
        if not color_frame:
            return None
        # 把 SDK 彩色帧转换成 OpenCV 可用的 (H,W,3) BGR NumPy 数组。
        return np.asanyarray(color_frame.get_data())

    def close(self) -> None:
        """停止 pipeline 并释放 RealSense。"""

        # close 可以重复调用；只有已启动时才真正 stop。
        if self._started:
            self._pipeline.stop()
            self._started = False


def parse_args() -> argparse.Namespace:
    """定义 RealSense、YOLO 推理和可视化参数。

    返回的 ``Namespace`` 可通过 ``args.serial``、``args.conf`` 等属性访问。
    """

    # 只取模块说明第一行作为简短 --help 标题。
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # 省略模型时会自动选择修改时间最新的 best.pt。
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "YOLO Seg 权重路径；省略时按修改时间选择 runs 下最新的 best.pt，"
            "有多个实验时建议显式指定。"
        ),
    )
    # 配置文件负责把四个模型类别组合成两种试管。
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "vision.yaml",
        help=(
            "二维几何配置；定义每种颜色的 body/cap 类别、显示颜色以及"
            "从管底到管盖的抓取比例。"
        ),
    )
    # 序列号是相机身份安全边界，多设备环境必须明确传入。
    parser.add_argument(
        "--serial",
        default=None,
        help="RealSense 设备序列号；检测到多台相机时必须显式指定。",
    )
    # 外部 eye-to-hand 是可选的；不提供时只显示图像二维几何。
    parser.add_argument(
        "--eye-to-hand",
        type=Path,
        default=None,
        help=(
            "可选：外部工程导出的 eye-to-hand YAML/JSON。必须采用 "
            "T_base_from_camera、米制单位并匹配当前彩色流；本脚本不求解标定。"
        ),
    )
    # 改平面高度意味着使用了标定验证范围外的组合，后面会降级状态。
    parser.add_argument(
        "--plane-z-m",
        type=float,
        default=None,
        help=(
            "可选：覆盖 eye-to-hand 文件中的固定抓取平面 Z，单位米。"
            "只对 --eye-to-hand 生效。"
        ),
    )
    # action="store_true"：命令行出现该开关时为 True，否则为 False。
    parser.add_argument(
        "--allow-unvalidated-eye-to-hand",
        action="store_true",
        help=(
            "只为坐标显示排查而允许外参或覆盖后的平面组合未能有效验证；"
            "画面会标记 UNVALIDATED，结果绝不能用于机械臂运动。"
        ),
    )
    # width/height/fps 是向 RealSense 请求的彩色流规格。
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="RealSense 彩色流宽度，单位为像素，默认 1280。",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="RealSense 彩色流高度，单位为像素，默认 720。",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="请求的 RealSense 彩色流帧率，单位为帧/秒，默认 30。",
    )
    # imgsz 是 YOLO 内部推理尺寸，并不会改变最终 B/C/G 使用的原帧坐标。
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="YOLO 推理缩放尺寸，不等同于相机分辨率，默认 1024。",
    )
    # conf 控制最低置信度；iou 控制重叠候选的非极大值抑制。
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="保留检测结果的最低置信度，范围 0..1，默认 0.25。",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="YOLO NMS 的 IoU 阈值，范围 0..1，默认 0.70。",
    )
    # device 可写 GPU 编号或 cpu。
    parser.add_argument(
        "--device",
        default="0",
        help="Ultralytics 推理设备，例如 0、1 或 cpu；默认使用 GPU 0。",
    )
    # max_det 限制每帧保留的实例数，为输出数量和后处理成本设置明确上限。
    parser.add_argument(
        "--max-det",
        type=int,
        default=50,
        help="每帧最多保留的 YOLO 实例数，默认 50。",
    )
    # line-width 只影响显示，不改变 mask 和几何计算。
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="分割结果检测框的绘制线宽，单位为像素，默认 2。",
    )
    # max-frames=0 表示无限循环；正数便于受控自动测试。
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="最多处理的彩色帧数；0 表示持续运行，无窗口测试时可设为 1 或 30。",
    )
    # no-display 只关闭 GUI，仍会真实打开相机并执行模型。
    parser.add_argument(
        "--no-display",
        action="store_true",
        help=(
            "不创建 OpenCV 窗口，但仍会打开 RealSense 并执行推理；"
            "配合 --max-frames 用于自动测试。"
        ),
    )
    # save-vis 是退出时保存的最后一帧完整面板。
    parser.add_argument(
        "--save-vis",
        type=Path,
        default=None,
        help="退出时保存最后一帧完整可视化；路径应包含图片文件名。",
    )
    # 默认不覆盖，以免新实验覆盖旧证据。
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的 --save-vis；默认拒绝覆盖。",
    )
    # snapshot-dir 用于运行中按 S 保存多张、带时间戳的截图。
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=ROOT / "runs" / "realtime_geometry" / "screenshots",
        help="有窗口运行时按 S 保存截图的目录。",
    )
    # 把 sys.argv 中的文本按 type 参数转换后返回。
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """在加载模型和打开相机前检查参数。

    参数错误应尽早失败：这样既不占用 GPU，也不会不必要地打开真实相机。
    """

    # 相机流规格必须全部为正整数。
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise ValueError("--width、--height 和 --fps 必须大于 0。")
    # 模型尺寸、最大实例数和绘图线宽同样不能为 0 或负数。
    if args.imgsz <= 0 or args.max_det <= 0 or args.line_width <= 0:
        raise ValueError("--imgsz、--max-det 和 --line-width 必须大于 0。")
    # isfinite 排除 NaN/无穷，再检查合法的 0..1 区间。
    if (
        not math.isfinite(args.conf)
        or not math.isfinite(args.iou)
        or not 0.0 <= args.conf <= 1.0
        or not 0.0 <= args.iou <= 1.0
    ):
        raise ValueError("--conf 和 --iou 必须是位于 0..1 的有限数值。")
    # 0 有“持续运行”的特殊含义，所以只拒绝负数。
    if args.max_frames < 0:
        raise ValueError("--max-frames 不能小于 0。")
    # 固定平面和允许未验证开关都依附于 eye-to-hand，不能单独出现。
    if args.plane_z_m is not None and args.eye_to_hand is None:
        raise ValueError("--plane-z-m 只能与 --eye-to-hand 一起使用。")
    if args.plane_z_m is not None and not math.isfinite(args.plane_z_m):
        raise ValueError("--plane-z-m 必须是有限数值。")
    if (
        args.allow_unvalidated_eye_to_hand
        and args.eye_to_hand is None
    ):
        raise ValueError(
            "--allow-unvalidated-eye-to-hand 只能与 --eye-to-hand 一起使用。"
        )
    # 在打开相机前做第一轮覆盖检查，避免运行结束才发现目标已存在。
    if (
        args.save_vis is not None
        and args.save_vis.expanduser().resolve().exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            "--save-vis 已存在，默认拒绝覆盖；请更换路径或显式添加 --overwrite："
            f"{args.save_vis.expanduser().resolve()}"
        )


def find_latest_best_weight() -> Path:
    """寻找 ``runs`` 中修改时间最新的 best.pt，而不是指标最高的实验。"""

    # rglob 递归搜索，兼容正常目录和旧版重复嵌套的 runs 目录。
    candidates: list[Path] = []
    for path in (ROOT / "runs").rglob("best.pt"):
        if path.is_file():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "没有在 runs 下找到 best.pt，请通过 --model 指定模型。"
        )
    # 逐个比较修改时间；“最新写入”不代表验证指标最高。
    latest_path = candidates[0]
    latest_time = latest_path.stat().st_mtime
    for path in candidates[1:]:
        modified_time = path.stat().st_mtime
        if modified_time > latest_time:
            latest_path = path
            latest_time = modified_time
    return latest_path.resolve()


def resolve_model_argument(value: str | None) -> str:
    """解析模型参数，同时允许 Ultralytics 模型名称。"""

    # 用户没有传 --model 时才自动查找本地最新权重。
    if value is None:
        return str(find_latest_best_weight())
    # expanduser 展开 ~；本地真实文件转换为绝对路径。
    candidate = Path(value).expanduser()
    # 不存在的字符串原样返回，因为它可能是 Ultralytics 模型名称。
    if candidate.is_file():
        return str(candidate.resolve())
    return value


def load_tube_config(path: Path) -> tuple[dict[str, TubeSpec], float]:
    """读取 vision.yaml 中的类别配对、颜色和抓取比例。"""

    # 统一加载器会严格检查结构、类别顺序、颜色范围和抓取比例。
    config = load_vision_config(path)
    specs: dict[str, TubeSpec] = {}
    for tube_name, spec in config.tubes.items():
        specs[tube_name] = TubeSpec(
            body_class=spec.body_class,
            cap_class=spec.cap_class,
            display_color_bgr=spec.display_color_bgr,
        )
    # 返回两个值：试管配置字典，以及 B→C 的抓取比例。
    return specs, config.grasp_fraction_from_bottom


def validate_model_classes(names: Any, specs: dict[str, TubeSpec]) -> None:
    """严格确认模型类别 ID/顺序与当前 ``vision.yaml`` 完全一致。"""

    # 按配置顺序展开 body/cap，得到模型应该具有的精确类别 ID 顺序。
    configured_names: list[str] = []
    for spec in specs.values():
        configured_names.append(spec.body_class)
        configured_names.append(spec.cap_class)
    configured = tuple(configured_names)
    validate_tube_model_contract(
        task="segment",
        names=names,
        configured_class_order=configured,
    )


def class_name_for_id(names: Any, class_id: int) -> str:
    """从 Ultralytics 类别字典或列表中取得类别名。"""

    # 当前 Ultralytics 的 names 可像映射一样通过整数 ID 索引。
    return str(names[int(class_id)])


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """把模型掩膜恢复到彩色帧尺寸，并转换成布尔数组。"""

    # NumPy shape 顺序是 (高, 宽)，OpenCV resize 尺寸参数是 (宽, 高)。
    if mask.shape == (height, width):
        return mask.astype(bool)
    # 最近邻插值不会像线性插值那样在二值边缘创造混合灰度。
    resized = cv2.resize(
        mask.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    # 大于 0.5 的预测像素变成前景 True，其余为 False。
    return resized > 0.5


def best_detections_by_name(
    result: Any,
    wanted_names: set[str],
    image_shape: tuple[int, int],
) -> dict[str, MaskDetection]:
    """一次读取推理结果，每个目标类别只返回置信度最高的一个掩膜。

    本函数只整理候选；调用方会先统计实例数。同色 body 或 cap 超过一个时，
    当前流程直接拒绝该颜色，不会把不同实例的最高分掩膜组合成几何结果。
    """

    # 没有 mask、box 或实例时直接返回空字典。
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return {}

    # detach→cpu→numpy 把推理张量从 GPU/梯度体系转成普通 NumPy 数据。
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy()
    masks = result.masks.data.detach().cpu().numpy()
    # 每个检测框必须有一张对应实例掩膜，否则下标配对已失效。
    if len(classes) != len(masks):
        raise RuntimeError(
            f"检测框数量 {len(classes)} 与掩膜数量 {len(masks)} 不一致。"
        )

    # 一次遍历所有预测，避免针对四个类别重复把整批 mask 从 GPU 搬到 CPU。
    # 值 tuple 保存 (结果下标, 当前最高置信度)。
    best_indices: dict[str, tuple[int, float]] = {}
    for index in range(len(classes)):
        class_id = classes[index]
        # 把 p-body 等类别字符串和该实例置信度取出。
        class_name = class_name_for_id(result.names, int(class_id))
        confidence = float(confidences[index])
        previous = best_indices.get(class_name)
        # 只关心 vision.yaml 声明的四类，并用更高分候选替换旧候选。
        if class_name in wanted_names and (
            previous is None or confidence > previous[1]
        ):
            best_indices[class_name] = (index, confidence)

    height, width = image_shape
    # 为每个目标类别生成一份已恢复原帧尺寸的 MaskDetection。
    detections: dict[str, MaskDetection] = {}
    for class_name, index_and_confidence in best_indices.items():
        index, confidence = index_and_confidence
        detections[class_name] = MaskDetection(
            mask=resize_mask(masks[index], width, height),
            confidence=confidence,
        )
    return detections


def calculate_frame_geometries(
    result: Any,
    image_shape: tuple[int, int],
    specs: dict[str, TubeSpec],
    grasp_fraction: float,
) -> dict[str, TubeFrameGeometry]:
    """计算当前帧中每种颜色最高置信度目标的二维图像几何。

    B（bottom）是远离管盖的一端，C（cap-side）是靠近管盖的“管身端点”，
    不是管盖 mask 中心；G（grasp）位于 B→C 的配置比例处。管盖缺失时只有
    无方向 E1/E2 和中点 G。
    """

    # set 去重得到本帧需要整理的四个模型类别。
    wanted_names: set[str] = set()
    for spec in specs.values():
        wanted_names.add(spec.body_class)
        wanted_names.add(spec.cap_class)
    detections = best_detections_by_name(result, wanted_names, image_shape)
    # 除了最高分候选，还必须统计实例总数来发现同色多目标歧义。
    if result.boxes is None:
        instance_counts: Counter[str] = Counter()
    else:
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        detected_names: list[str] = []
        for class_id in class_ids:
            detected_names.append(class_name_for_id(result.names, class_id))
        instance_counts = Counter(detected_names)

    # 每种颜色无论成功还是失败都会放入一条 TubeFrameGeometry。
    geometries: dict[str, TubeFrameGeometry] = {}
    for tube_name, spec in specs.items():
        body_count = instance_counts.get(spec.body_class, 0)
        cap_count = instance_counts.get(spec.cap_class, 0)
        if body_count > 1 or cap_count > 1:
            # 不能把两支同色试管中各自最高分的 body/cap 随意组合。
            cap_confidence = None
            if cap is not None:
                cap_confidence = cap.confidence
            geometries[tube_name] = TubeFrameGeometry(
                tube_name=tube_name,
                spec=spec,
                pose=None,
                body_confidence=None,
                cap_confidence=None,
                error=(
                    "ambiguous same-color instances: "
                    f"{spec.body_class}={body_count}, "
                    f"{spec.cap_class}={cap_count}"
                ),
            )
            continue
        # 单实例场景再取对应管身与管盖；管盖允许缺失，管身不允许。
        body = detections.get(spec.body_class)
        cap = detections.get(spec.cap_class)
        if body is None:
            # 没有管身就没有足够形状信息做 PCA 长轴。
            geometries[tube_name] = TubeFrameGeometry(
                tube_name=tube_name,
                spec=spec,
                pose=None,
                body_confidence=None,
                cap_confidence=cap_confidence,
                error=f"missing {spec.body_class}",
            )
            continue

        try:
            # 几何库会保留管身最大连通域，用初次 PCA 投影裁掉两端各 5% 的
            # 离群像素后重新拟合，并检查主轴长度及 SVD 奇异值比（不是外接框
            # 长宽比）。cap 中心还必须靠近长轴且明确靠近某一端，才可判定 C。
            cap_mask = None
            selected_grasp_fraction = 0.5
            if cap is not None:
                cap_mask = cap.mask
                selected_grasp_fraction = grasp_fraction
            pose = tube_pose_from_masks(
                body.mask,
                cap_mask,
                # 没有管盖时无法判断从哪一端量抓取比例，因此只使用无方向中点。
                selected_grasp_fraction,
            )
            # 无 cap 时 pose 仍包含无方向轴，但保留错误文本提醒朝向未解决。
            error = None
            if cap is None:
                error = f"missing {spec.cap_class}"
        except ValueError as exc:
            # 掩膜过短、接近圆形或为空等质量问题均不会产生可用 pose。
            pose = None
            error = str(exc)

        # 把本颜色的检测置信度、姿态和错误统一装进不可变数据对象。
        cap_confidence = None
        if cap is not None:
            cap_confidence = cap.confidence
        geometries[tube_name] = TubeFrameGeometry(
            tube_name=tube_name,
            spec=spec,
            pose=pose,
            body_confidence=body.confidence,
            cap_confidence=cap_confidence,
            error=error,
        )
    return geometries


def apply_eye_to_hand(
    geometries: dict[str, TubeFrameGeometry],
    calibration: ExternalEyeToHandCalibration,
    plane_z_m: float | None,
) -> dict[str, TubeFrameGeometry]:
    """把每支试管的 B/C/G 投影到 CArm 基座固定平面。

    单个目标转换失败只会在该目标上记录 ``robot_error``，不会把一个危险或
    非有限坐标留在结果中。没有管盖时仍可显示 G，但不会输出有方向的 yaw。
    """

    # 不原地修改输入字典，创建一份带机器人坐标的新结果。
    converted: dict[str, TubeFrameGeometry] = {}
    for tube_name, geometry in geometries.items():
        # 没有二维姿态就没有可反投影的 G，原样保留错误记录。
        if geometry.pose is None:
            converted[tube_name] = geometry
            continue

        pose = geometry.pose
        try:
            # 像素 G 先按内参和畸变模型变为相机射线；外参的旋转用于射线
            # 方向、平移用于基座中的相机光心，最后与 z=plane_z_m 求交。
            # 本脚本不读取深度，这一步明确假设 B/C/G 位于同一固定 Z 平面。
            grasp_xyz = calibration.pixel_to_base_plane(
                pose.grasp_xy,
                plane_z_m=plane_z_m,
            )
            if geometry.direction_resolved:
                # cap 存在时同时转换 B/C/G，保证三点使用同一转换约定。
                bcg = calibration.bcg_to_base(
                    pose.bottom_xy,
                    pose.cap_xy,
                    pose.grasp_xy,
                    plane_z_m=plane_z_m,
                )
                bottom_xyz = bcg["B"]
                cap_xyz = bcg["C"]
                # 基座平面中的 yaw 来自 B→C 向量，不是图像角度直接复制。
                yaw_rad = math.atan2(
                    cap_xyz[1] - bottom_xyz[1],
                    cap_xyz[0] - bottom_xyz[0],
                )
            else:
                # 无管盖时 PCA 正负方向任意，因此不输出 B/C 和 yaw。
                bottom_xyz = None
                cap_xyz = None
                yaw_rad = None
            # dataclasses.replace 返回一个新对象，只更新列出的机器人字段。
            converted[tube_name] = replace(
                geometry,
                robot_grasp_xyz_m=grasp_xyz,
                robot_bottom_xyz_m=bottom_xyz,
                robot_cap_xyz_m=cap_xyz,
                robot_yaw_rad=yaw_rad,
                robot_error=None,
            )
        except ValueError as exc:
            # 单目标射线近平行、交点过远等失败只标记该目标，不终止整个预览。
            converted[tube_name] = replace(
                geometry,
                robot_grasp_xyz_m=None,
                robot_bottom_xyz_m=None,
                robot_cap_xyz_m=None,
                robot_yaw_rad=None,
                robot_error=str(exc),
            )
    return converted


def draw_point_label(
    image: np.ndarray,
    point_xy: tuple[int, int],
    label: str,
    color_bgr: tuple[int, int, int],
) -> None:
    """绘制几何点及其单字母标签。"""

    # thickness=-1 表示实心圆；LINE_AA 表示抗锯齿边缘。
    cv2.circle(image, point_xy, 7, color_bgr, -1, cv2.LINE_AA)
    cv2.putText(
        image,
        label,
        (point_xy[0] + 8, point_xy[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color_bgr,
        2,
        cv2.LINE_AA,
    )


def draw_frame_geometries(
    image: np.ndarray,
    geometries: dict[str, TubeFrameGeometry],
) -> np.ndarray:
    """在分割画面上叠加长轴、B/C/G 点和角度。"""

    # copy 避免原地修改调用者给出的 Ultralytics 可视化数组。
    output = image.copy()
    for geometry in geometries.values():
        # 检测/质量失败的目标没有可画姿态。
        if geometry.pose is None:
            continue

        pose = geometry.pose
        # OpenCV 绘图需要整数像素；几何计算保留浮点精度到最后才取整。
        bottom = tuple(np.round(pose.bottom_xy).astype(int))
        cap = tuple(np.round(pose.cap_xy).astype(int))
        grasp = tuple(np.round(pose.grasp_xy).astype(int))
        axis_color = geometry.spec.display_color_bgr

        # 直线表示管身 PCA 长轴。
        cv2.line(output, bottom, cap, axis_color, 3, cv2.LINE_AA)
        if geometry.direction_resolved:
            draw_point_label(output, bottom, "B", (255, 0, 0))
            draw_point_label(output, cap, "C", (0, 0, 255))
        else:
            # 管盖缺失时 PCA 长轴的符号是任意的，不能把两端称为管底和管盖。
            draw_point_label(output, bottom, "E1", (180, 180, 180))
            draw_point_label(output, cap, "E2", (180, 180, 180))
        # G 始终是可显示的抓取候选；无 cap 时它被强制放在中点。
        draw_point_label(output, grasp, "G", (0, 255, 255))

        if geometry.direction_resolved:
            # 图像坐标 y 轴向下，因此正角度在画面中表现为顺时针。
            angle_text = f"{math.degrees(pose.angle_rad):.1f} deg"
        else:
            # PCA 长轴没有方向，使用模 180 度的角度，避免逐帧出现 180 度跳变。
            axis_angle_deg = math.degrees(pose.angle_rad) % 180.0
            angle_text = f"axis {axis_angle_deg:.1f} deg mod 180"
        label = f"{geometry.tube_name} {angle_text}"
        label_origin = (grasp[0] + 14, grasp[1] + 22)
        cv2.putText(
            output,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            axis_color,
            2,
            cv2.LINE_AA,
        )
        # 只有 eye-to-hand 转换成功才叠加基座坐标。
        if geometry.robot_grasp_xyz_m is not None:
            x_m, y_m, z_m = geometry.robot_grasp_xyz_m
            robot_text = f"G base=({x_m:.3f},{y_m:.3f},{z_m:.3f})m"
            if geometry.robot_yaw_rad is not None:
                # yaw 只有 cap 消除长轴方向歧义后才存在。
                robot_text += (
                    f" yaw={math.degrees(geometry.robot_yaw_rad):.1f}deg"
                )
            cv2.putText(
                output,
                robot_text,
                (grasp[0] + 14, grasp[1] + 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (80, 255, 80),
                2,
                cv2.LINE_AA,
            )
    return output


def class_count_text(result: Any) -> str:
    """把每帧检测到的类别数量整理为一行文本。"""

    if result.boxes is None or len(result.boxes) == 0:
        return "objects: none"
    # 把 GPU 类别张量转为整数数组，再用 Counter 计数。
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    detected_names: list[str] = []
    for class_id in class_ids:
        detected_names.append(class_name_for_id(result.names, class_id))
    counts = Counter(detected_names)
    # sorted 让类别显示顺序稳定，避免每帧文本次序跳动。
    count_parts: list[str] = []
    for name in sorted(counts):
        count_parts.append(f"{name}:{counts[name]}")
    return " | ".join(count_parts)


def geometry_status_lines(
    geometries: dict[str, TubeFrameGeometry],
) -> list[str]:
    """生成紫色、黄色试管的实时几何状态文本。"""

    lines: list[str] = []
    for tube_name, geometry in geometries.items():
        # 没有 pose 时直接显示具体失败原因。
        if geometry.pose is None:
            lines.append(f"{tube_name}: {geometry.error or 'not detected'}")
            continue

        # 先显示 --；存在置信度时再替换成两位小数。
        body_text = "--"
        if geometry.body_confidence is not None:
            body_text = f"{geometry.body_confidence:.2f}"
        cap_text = "--"
        if geometry.cap_confidence is not None:
            cap_text = f"{geometry.cap_confidence:.2f}"
        # 无管盖时刻意不显示有方向角，防止用户误解为可抓取 yaw。
        angle_text = "direction unresolved"
        if geometry.direction_resolved:
            angle_text = f"{math.degrees(geometry.pose.angle_rad):.1f} deg"
        # 长轴像素长度和 PCA 奇异值比帮助判断几何是否接近质量下限。
        length_text = "--"
        if geometry.pose.length_px is not None:
            length_text = f"{geometry.pose.length_px:.1f}px"
        aspect_text = "--"
        if geometry.pose.pca_aspect_ratio is not None:
            aspect_text = f"{geometry.pose.pca_aspect_ratio:.2f}"
        lines.append(
            f"{tube_name}: body={body_text} cap={cap_text} "
            f"len={length_text} aspect={aspect_text} angle={angle_text}"
        )
        if geometry.robot_error:
            # 转换错误优先于坐标，避免继续展示上一次或不完整结果。
            lines.append(f"{tube_name} base: ERROR {geometry.robot_error}")
        elif geometry.robot_grasp_xyz_m is not None:
            x_m, y_m, z_m = geometry.robot_grasp_xyz_m
            yaw_text = "unresolved"
            if geometry.robot_yaw_rad is not None:
                yaw_text = f"{math.degrees(geometry.robot_yaw_rad):.1f} deg"
            lines.append(
                f"{tube_name} base: G=({x_m:.4f},{y_m:.4f},{z_m:.4f}) m "
                f"yaw={yaw_text}"
            )
    return lines


def draw_status_panel(
    image: np.ndarray,
    fps: float,
    result: Any,
    geometries: dict[str, TubeFrameGeometry],
    model_name: str,
    calibration_status: str | None,
    show_boxes: bool,
    show_labels: bool,
    show_geometry: bool,
    paused: bool,
) -> np.ndarray:
    """在画面底部绘制帧率、检测数量和几何状态。"""

    # Ultralytics 在 result.speed 中记录单帧推理毫秒数；缺字段时显示 0。
    inference_ms = float(result.speed.get("inference", 0.0))
    # 面板内容先在 Python 列表中组织，再统一计算背景高度。
    lines = [
        f"model: {model_name}",
        f"FPS: {fps:.1f} | inference: {inference_ms:.1f} ms",
        class_count_text(result),
    ]
    if calibration_status is not None:
        lines.append(calibration_status)
    # 用户按 G 隐藏几何时，也隐藏几何状态行。
    if show_geometry:
        lines.extend(geometry_status_lines(geometries))
    boxes_text = "off"
    if show_boxes:
        boxes_text = "on"
    labels_text = "off"
    if show_labels:
        labels_text = "on"
    geometry_text = "off"
    if show_geometry:
        geometry_text = "on"
    paused_text = ""
    if paused:
        paused_text = " | PAUSED"
    lines.extend(
        [
            f"boxes:{boxes_text} labels:{labels_text} "
            f"geometry:{geometry_text}{paused_text}",
            "Q quit | P/SPACE pause | S snapshot | B boxes | L labels | G geometry",
        ]
    )

    # 根据文字行数动态计算半透明黑色面板大小。
    line_height = 24
    panel_height = 14 + line_height * len(lines)
    panel_width = min(image.shape[1], 790)
    panel_top = max(image.shape[0] - panel_height, 0)
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (0, panel_top),
        (panel_width, image.shape[0]),
        (0, 0, 0),
        -1,
    )
    # addWeighted 把黑色 overlay 与原图混合，保证白字在复杂背景上可读。
    cv2.addWeighted(overlay, 0.64, image, 0.36, 0.0, image)

    # enumerate 提供从 0 开始的行号，用于计算每行 y 坐标。
    for index in range(len(lines)):
        text = lines[index]
        cv2.putText(
            image,
            text,
            (12, panel_top + 24 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image


def save_snapshot(image: np.ndarray, output_dir: Path) -> Path:
    """保存当前实时几何画面。"""

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # 微秒也写入文件名，降低连续按 S 时重名覆盖的概率。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"geometry_{timestamp}.jpg"
    # cv2.imwrite 失败常用 False 表示，必须主动检查返回值。
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"截图保存失败：{output_path}")
    return output_path


def write_image_atomic(
    path: Path,
    image: np.ndarray,
    *,
    overwrite: bool,
) -> None:
    """暂存图片后原子发布；默认使用硬链接保证不覆盖已有文件。

    原子发布保证其他进程不会读到只写了一部分的图片；默认 ``os.link`` 在
    目标已存在时失败，从操作系统层面消除“先检查、后覆盖”的竞争窗口。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    # OpenCV 根据扩展名选择编码器；没有扩展名时临时文件默认使用 PNG。
    suffix = path.suffix
    if not suffix:
        suffix = ".png"
    # mkstemp 安全创建同目录临时文件，返回底层描述符和唯一文件名。
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=suffix,
    )
    # imwrite 通过路径重新打开文件，因此先关闭 mkstemp 的描述符。
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        if not cv2.imwrite(str(temporary_path), image):
            raise RuntimeError(f"最后一帧保存失败：{path}")
        if overwrite:
            # 明确 --overwrite 时允许原子替换目标。
            os.replace(temporary_path, path)
        else:
            # 硬链接要求目标不存在；若运行期间出现同名文件，这里安全失败。
            os.link(temporary_path, path)
    finally:
        # replace/link 后临时名字不再需要；失败时也清理垃圾文件。
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    """加载模型、打开 RealSense 并进入实时二维几何预览循环。"""

    # 先解析和校验纯参数，错误时不加载模型、不占用相机。
    args = parse_args()
    validate_args(args)
    # 用户没有提供 --eye-to-hand 时保持 None，整个脚本只计算图像二维几何。
    eye_to_hand = None
    if args.eye_to_hand:
        eye_to_hand = ExternalEyeToHandCalibration.load(args.eye_to_hand)
    # 这两个布尔量区分“外部文件的 validation 元数据声明 validated”与
    # “当前平面没有改变”。本脚本信任并校验元数据结构，不重新求解标定。
    plane_override_changed = False
    eye_to_hand_effectively_validated = False
    if eye_to_hand is not None:
        # 当前像素来自彩色流，不能套用 depth_optical 等其他坐标系外参。
        if eye_to_hand.camera_frame != "color_optical":
            raise ValueError(
                "本脚本读取 RealSense 彩色流，因此 eye-to-hand 的 camera.frame "
                f"必须是 color_optical，当前为 {eye_to_hand.camera_frame!r}。"
            )
        # 绝对误差 1e-9 m 内视作相同，避免浮点表示噪声触发误判。
        plane_override_changed = (
            args.plane_z_m is not None
            and not math.isclose(
                float(args.plane_z_m),
                eye_to_hand.plane_z_m,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
        # 命令行改过平面后，外部工程原先的独立验证不覆盖这个新组合。
        eye_to_hand_effectively_validated = (
            eye_to_hand.validation.is_validated and not plane_override_changed
        )
        # 默认安全策略是验证不完整就终止；只能用显式开关降级为 DISPLAY ONLY。
        # 这个开关不会放过已知的相机序列号、分辨率、坐标系或内参不匹配。
        if (
            not eye_to_hand_effectively_validated
            and not args.allow_unvalidated_eye_to_hand
        ):
            reason = "validation.status 不是 validated"
            if plane_override_changed:
                reason = "命令行覆盖的 plane_z_m 没有包含在外部独立验证中"
            raise ValueError(
                f"eye-to-hand 当前不能视为已验证：{reason}。"
                "若只做离线式画面排查，可显式添加 "
                "--allow-unvalidated-eye-to-hand；不得把结果用于机械臂运动。"
            )

    # 延迟导入，使未安装 Ultralytics 时仍能查看 --help。
    from ultralytics import YOLO

    # 读取类别配对/抓取比例，解析模型路径并加载 YOLO 权重。
    tube_specs, grasp_fraction = load_tube_config(args.config)
    model_argument = resolve_model_argument(args.model)
    print(f"Model: {model_argument}")
    model = YOLO(model_argument)
    # 展开后应为 (p-body, p-cap, y-body, y-cap) 的严格 ID 顺序。
    configured_class_names: list[str] = []
    for spec in tube_specs.values():
        configured_class_names.append(spec.body_class)
        configured_class_names.append(spec.cap_class)
    configured_class_order = tuple(configured_class_names)
    # 在创建 RealSenseColorSource（即打开相机）之前完成任务及类别 ID 校验。
    validate_tube_model_contract(
        task=model.task,
        names=model.names,
        configured_class_order=configured_class_order,
    )
    print(f"Classes: {model.names}")
    print(f"Grasp fraction from bottom: {grasp_fraction:.2f}")

    # 只有模型任务和类别验证全部通过后才真正打开 RealSense。
    requested_serial = None
    if args.serial:
        requested_serial = str(args.serial).strip()
    source = RealSenseColorSource(
        args.width,
        args.height,
        args.fps,
        requested_serial,
    )
    if eye_to_hand is not None:
        try:
            # 用“实际启动后的”序列号、分辨率和内参校验外部标定，
            # 不把命令行请求值当成已经发生的事实。内参检查覆盖焦距、主点、
            # 畸变模型和有效系数；任何已知不匹配都会终止，即使开启降级显示。
            eye_to_hand.validate_stream(
                serial=source.serial,
                width=source.width,
                height=source.height,
                camera_frame="color_optical",
            )
            eye_to_hand.validate_intrinsics(
                fx=source.intrinsics["fx"],
                fy=source.intrinsics["fy"],
                ppx=source.intrinsics["ppx"],
                ppy=source.intrinsics["ppy"],
                distortion_model=source.intrinsics["distortion_model"],
                coeffs=source.intrinsics["coeffs"],
            )
        except Exception:
            # 标定校验失败必须立即释放相机，再继续抛出错误。
            source.close()
            raise
    print(f"Source: {source.description}")
    if eye_to_hand is not None:
        # 终端清楚区分可验证坐标与仅供显示的降级坐标。
        validation_word = "UNVALIDATED - DISPLAY ONLY"
        if eye_to_hand_effectively_validated:
            validation_word = "VALIDATED"
        # 若没有覆盖，使用标定文件里的固定工作平面 Z。
        plane_z = eye_to_hand.plane_z_m
        if args.plane_z_m is not None:
            plane_z = float(args.plane_z_m)
        print(
            "Eye-to-hand: "
            f"{validation_word}, p_base=T_base_from_camera@p_camera, "
            f"plane_z={plane_z:.6f} m"
        )
    print("Keys: Q/ESC quit, P/SPACE pause, S snapshot, B boxes, L labels, G geometry")
    # 同一状态也会放到每帧画面底部，防止截图脱离终端后丢失安全语义。
    calibration_status = None
    if eye_to_hand is not None:
        calibration_status = "eye-to-hand: UNVALIDATED / DISPLAY ONLY"
        if eye_to_hand_effectively_validated:
            calibration_status = "eye-to-hand: VALIDATED"

    # last_base_visualization：按当前显示选项生成、但还没有底部状态面板；
    # last_visualization：包含状态面板，适合显示和退出保存。
    last_base_visualization: np.ndarray | None = None
    last_visualization: np.ndarray | None = None
    # 暂停时需要复用上一帧结果，所以保留 result 和 geometries。
    last_result: Any | None = None
    last_geometries: dict[str, TubeFrameGeometry] = {}
    # 以下布尔状态由 B/L/G/P 按键切换。
    show_boxes = True
    show_labels = True
    show_geometry = True
    paused = False
    # 第一次 model.predict 通常包含 CUDA 初始化等额外开销，先做一次热身。
    warmed_up = False
    processed_frames = 0
    # EMA 帧率以实际启动 profile 的 fps 作为初值，后面逐帧平滑更新。
    # 它衡量相邻“整帧处理完成时刻”的端到端吞吐率，不等同于相机硬件帧率，
    # 也不等同于 result.speed 中仅模型 inference 阶段的耗时。
    fps_ema = float(source.fps)
    last_frame_completed_at: float | None = None
    window_name = "HPS RealSense realtime geometry"

    if not args.no_display:
        try:
            # WINDOW_NORMAL 允许用户拖动窗口改变显示大小。
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except Exception:
            # GUI 创建失败时同样不能遗留被占用的相机。
            source.close()
            raise

    try:
        # 实时循环直到 Q/Esc、max_frames 或 Ctrl+C。
        while True:
            # 暂停时不读取新帧、不做新推理，只继续处理键盘和重复显示上一帧。
            if not paused:
                frame = source.read()
                if frame is None:
                    # 该 frameset 没有彩色帧，跳过而不是给模型传 None。
                    continue

                if not warmed_up:
                    # 热身结果故意丢弃，不计入 processed_frames。
                    model.predict(
                        source=frame,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        iou=args.iou,
                        device=args.device,
                        # 项目视觉脚本统一固定使用 FP32。
                        half=False,
                        max_det=args.max_det,
                        retina_masks=True,
                        verbose=False,
                    )
                    warmed_up = True

                # 单帧输入返回长度为 1 的结果列表，取第一个元素。
                prediction_results = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    half=False,
                    max_det=args.max_det,
                    retina_masks=True,
                    verbose=False,
                )
                result = prediction_results[0]
                height, width = frame.shape[:2]
                # 先计算二维图像几何；提供外部标定时，再做固定平面机器人坐标转换。
                geometries = calculate_frame_geometries(
                    result,
                    (height, width),
                    tube_specs,
                    grasp_fraction,
                )
                if eye_to_hand is not None:
                    # 此转换只生成/显示坐标，绝不发送任何机械臂命令。
                    geometries = apply_eye_to_hand(
                        geometries,
                        eye_to_hand,
                        args.plane_z_m,
                    )

                # Ultralytics 先绘制实例 mask、框和标签。
                visualization = result.plot(
                    conf=True,
                    line_width=args.line_width,
                    labels=show_labels,
                    boxes=show_boxes,
                    masks=True,
                    color_mode="class",
                )
                if show_geometry:
                    # 再叠加本项目计算的 B/C/G、长轴和可选基座坐标。
                    visualization = draw_frame_geometries(
                        visualization,
                        geometries,
                    )

                # perf_counter 是单调高精度时钟，适合计算耗时，不受系统时间校准影响。
                frame_completed_at = time.perf_counter()
                if last_frame_completed_at is not None:
                    # 1e-9 防止极端情况下除以 0。
                    elapsed = max(frame_completed_at - last_frame_completed_at, 1e-9)
                    instant_fps = 1.0 / elapsed
                    # 指数滑动平均减少单帧抖动：88% 历史 + 12% 新测量。
                    # 暂停期间时间也进入恢复后的第一次间隔，因此刚继续时 FPS
                    # 可能短暂下降，随后会由新帧逐步恢复。
                    fps_ema = 0.88 * fps_ema + 0.12 * instant_fps
                last_frame_completed_at = frame_completed_at

                # 保存这一帧完整状态，以供暂停、按键截图和退出保存复用。
                last_base_visualization = visualization
                last_result = result
                last_geometries = geometries
                last_visualization = draw_status_panel(
                    visualization.copy(),
                    fps_ema,
                    result,
                    geometries,
                    Path(model_argument).name,
                    calibration_status,
                    show_boxes,
                    show_labels,
                    show_geometry,
                    paused=False,
                )
                processed_frames += 1

            # 启动后若还没有任何有效帧，就不能显示、截图或判断按键状态。
            if (
                last_base_visualization is None
                or last_visualization is None
                or last_result is None
            ):
                continue

            if not args.no_display:
                if paused:
                    # 暂停状态重新绘制面板，让画面明确显示 PAUSED。
                    shown = draw_status_panel(
                        last_base_visualization.copy(),
                        fps_ema,
                        last_result,
                        last_geometries,
                        Path(model_argument).name,
                        calibration_status,
                        show_boxes,
                        show_labels,
                        show_geometry,
                        paused=True,
                    )
                else:
                    shown = last_visualization
                cv2.imshow(window_name, shown)
                # paused 时等待 30ms，减少静止画面无意义的 CPU 占用；
                # & 0xFF 兼容不同平台 waitKey 的高位返回差异。
                wait_milliseconds = 1
                if paused:
                    wait_milliseconds = 30
                key = cv2.waitKey(wait_milliseconds) & 0xFF
            else:
                # 255 表示“没有按键”；无窗口模式依靠 max_frames 或 Ctrl+C 结束。
                key = 255

            # Esc 的键码是 27；ord 把字符转换成对应整数键码。
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("p"), ord("P"), ord(" ")):
                # not 实现布尔开关切换。
                paused = not paused
            elif key in (ord("s"), ord("S")):
                # 截图使用最近一次已完成帧的状态面板；暂停时它不包含后来重绘的
                # “PAUSED”字样，因为 last_visualization 保存的是完成帧版本。
                snapshot_path = save_snapshot(last_visualization, args.snapshot_dir)
                print(f"Snapshot: {snapshot_path}")
            elif key in (ord("b"), ord("B")):
                show_boxes = not show_boxes
            elif key in (ord("l"), ord("L")):
                show_labels = not show_labels
            elif key in (ord("g"), ord("G")):
                show_geometry = not show_geometry

            # max_frames=0 时此条件永远不成立；正数达到上限后受控退出。
            if args.max_frames > 0 and processed_frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        # Ctrl+C 是用户正常终止方式，打印说明后仍进入 finally 清理资源。
        print("Stopped by Ctrl+C.")
    finally:
        try:
            # 无论推理、绘图、按键处理哪里抛异常，都必须关闭相机。
            source.close()
        finally:
            if not args.no_display:
                # 相机关闭即使异常，仍尝试销毁 OpenCV 窗口。
                cv2.destroyAllWindows()

    if args.save_vis is not None and last_visualization is not None:
        # 运行期间可能有其他进程新建目标，发布前再做第二轮覆盖检查。
        save_path = args.save_vis.expanduser().resolve()
        if save_path.exists() and not args.overwrite:
            raise FileExistsError(
                "--save-vis 在运行期间已出现，默认拒绝覆盖；"
                f"请更换路径或显式添加 --overwrite：{save_path}"
            )
        # 原子保存完整最后一帧，默认不覆盖旧结果。
        write_image_atomic(
            save_path,
            last_visualization,
            overwrite=args.overwrite,
        )
        print(f"Visualization: {save_path}")

    print(f"Processed frames: {processed_frames}")
    if fps_ema > 0.0:
        print(f"Smoothed FPS: {fps_ema:.2f}")
    # 0 作为正常退出码交给 SystemExit。
    return 0


# 直接运行脚本才进入 main；测试 import 本模块时不会自动打开相机。
if __name__ == "__main__":
    raise SystemExit(main())
