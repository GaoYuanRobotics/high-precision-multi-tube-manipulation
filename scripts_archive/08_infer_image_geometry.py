#!/usr/bin/env python3
"""第 08 步：对单张图片执行分割并计算紫色、黄色试管的二维几何信息。

当前模型包含四个类别：

- ``p-body``（管身）+ ``p-cap``（盖子）：紫色试管；
- ``y-body``（管身）+ ``y-cap``（盖子）：黄色试管。

脚本会在每种颜色只有一个管身和一个盖子候选时计算中心、管底、盖子端、
抓取点、从管底指向盖子的单位长轴及图像角度。若同时提供旧像素到机器人平面
二维仿射 JSON，或外部工程生成的 ``T_base_from_camera`` eye-to-hand 标定
文件，还会输出机器人坐标系下的位置和 yaw。旧二维仿射只供历史回归，正式
候选流程只接受外部 eye-to-hand 结果。本脚本不包含手眼标定求解，也不会连接
或控制机械臂。

同一颜色出现多个管身或多个盖子时，由于尚未实现可靠的实例空间配对，脚本会
拒绝输出该颜色的几何结果。因此本步骤适用于画面中至多一支紫色和一支黄色
试管的当前实验。盖子掩膜用于消除 PCA 长轴的 180 度方向歧义；
未检测到盖子时仍可估计无方向长轴，但只输出 ``E1/E2``、中点 G 和模 180°
轴角，不声明管底/盖子，也不输出机器人抓取 yaw。

几何库默认要求管身主轴至少 30 px、PCA 长短轴比至少 2.0；不满足时会把该
颜色标为几何质量失败。成功结果同时记录 ``length_px`` 和
``pca_aspect_ratio``，便于离线复核。

如果省略 ``--model``，脚本会递归搜索项目 ``runs`` 目录并自动选择修改时间
最新的 ``best.pt``，兼容当前标准目录和早期重复嵌套的训练目录。当前四类模型
没有 ``rack_top``，因此本脚本不计算试管架或孔位。

典型运行命令：

    python scripts/08_infer_image_geometry.py \
      --image data/raw/20260720_114924/color/frame_000000.jpg \
      --config configs/vision.yaml \
      --imgsz 1024 \
      --conf 0.25 \
      --save-vis runs/geometry/frame_000000.jpg
"""

from __future__ import annotations

# argparse：读取命令行中的 --image、--model 等参数。
import argparse
# json：把计算结果编码成便于程序读取和人工检查的 JSON 文本。
import json
# math：角度转换、atan2 以及有限数值检查。
import math
# os：提供 fsync、replace 和 link，用于安全发布输出文件。
import os
# tempfile：先写临时文件，成功后再原子发布到最终路径。
import tempfile
# dataclass：用结构明确、不可变的数据类保存检测和相机身份。
from dataclasses import dataclass
# Path：以面向对象方式拼接、规范化和检查文件路径。
from pathlib import Path
# Any：Ultralytics 的结果对象等第三方动态对象没有固定静态类型。
from typing import Any

# OpenCV 负责读取图片、调整掩膜大小以及绘制/保存可视化。
import cv2
# NumPy 负责掩膜数组、二维点和数值计算。
import numpy as np

# __file__ 是本脚本路径；parents[0] 是 scripts，parents[1] 是项目根目录。
# 自动找权重、默认配置和导入项目模块时均以这个根目录为基准。
ROOT = Path(__file__).resolve().parents[1]
# =============================================================================
# 本脚本自带的视觉几何和标定实现
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

"""旧版二维像素到机械臂平面的仿射标定函数。"""


import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Sequence

import numpy as np


def _pixel_unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """拒绝任意层级的 JSON 重复键。"""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate calibration JSON key: {key!r}.")
        result[key] = value
    return result


@dataclass(frozen=True)
class PixelRobotAffine:
    """保存图像像素 (u,v) 到机械臂平面 (x,y) 的旧版仿射变换。"""

    matrix_2x3: np.ndarray

    def __post_init__(self) -> None:
        # 必须复制并设为只读；否则调用方可在通过有限值校验后修改原数组，
        # 使这个冻结 dataclass 内部出现 NaN 或另一套未验证矩阵。
        """对象创建后立即校验字段形状、单位和有限值。"""

        matrix = np.array(self.matrix_2x3, dtype=np.float64, copy=True)
        if matrix.shape != (2, 3):
            raise ValueError("Affine matrix must have shape (2, 3).")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Affine matrix cannot contain NaN or infinity.")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix_2x3", matrix)

    def transform_point(self, uv: Sequence[float]) -> tuple[float, float]:
        """把一个二维像素点转换到目标二维坐标系。"""

        if len(uv) != 2:
            raise ValueError("Pixel point must contain exactly (u, v).")
        u, v = float(uv[0]), float(uv[1])
        if not np.all(np.isfinite([u, v])):
            raise ValueError("Pixel point cannot contain NaN or infinity.")
        # [x, y] = 2x3矩阵 @ [u, v, 1]，这是俯视场景里最简单稳的标定模型。
        with np.errstate(over="ignore", invalid="ignore"):
            xy = self.matrix_2x3 @ np.array(
                [u, v, 1.0],
                dtype=np.float64,
            )
        if not np.all(np.isfinite(xy)):
            raise ValueError("Affine transform produced NaN or infinity.")
        return float(xy[0]), float(xy[1])

    def transform_points(self, uv_points: Sequence[Sequence[float]]) -> np.ndarray:
        """批量转换多个二维像素点。"""

        points = np.asarray(uv_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Pixel points must have shape (N, 2).")
        if not np.all(np.isfinite(points)):
            raise ValueError("Pixel points cannot contain NaN or infinity.")
        ones = np.ones((points.shape[0], 1), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            transformed = (
                np.column_stack([points, ones]) @ self.matrix_2x3.T
            )
        if not np.all(np.isfinite(transformed)):
            raise ValueError("Affine transform produced NaN or infinity.")
        return transformed

    def save(self, path: Path, *, overwrite: bool = False) -> None:
        """原子保存旧仿射文件；默认不覆盖已有结果。"""

        data = {
            "type": "pixel_to_robot_affine",
            "matrix_2x3": self.matrix_2x3.tolist(),
        }
        target = path.expanduser().absolute()
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            if overwrite:
                os.replace(temporary, target)
            else:
                try:
                    os.link(temporary, target)
                except FileExistsError as exc:
                    raise FileExistsError(
                        f"Calibration output already exists: {target}. "
                        "Use explicit overwrite only after checking the path."
                    ) from exc
                temporary.unlink()
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> "PixelRobotAffine":
        """从磁盘读取并严格校验当前类型的数据文件。"""

        source = path.expanduser().resolve()
        data = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_pixel_unique_json_object,
        )
        if not isinstance(data, dict):
            raise ValueError("Calibration JSON top level must be an object.")
        unknown = set(data) - {"type", "matrix_2x3"}
        if unknown:
            raise ValueError(
                f"Calibration JSON contains unknown fields: {sorted(unknown)}."
            )
        if data.get("type") != "pixel_to_robot_affine":
            raise ValueError(
                "Calibration type must be 'pixel_to_robot_affine'."
            )
        if "matrix_2x3" not in data:
            raise ValueError("Calibration JSON is missing matrix_2x3.")
        return cls(np.asarray(data["matrix_2x3"], dtype=np.float64))


def fit_affine(pixel_uv: Sequence[Sequence[float]], robot_xy: Sequence[Sequence[float]]) -> PixelRobotAffine:
    """用至少三组对应点拟合最小二乘二维仿射变换。"""

    try:
        pixels = np.asarray(pixel_uv, dtype=np.float64)
        robots = np.asarray(robot_xy, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("pixel_uv and robot_xy must contain numeric values.") from exc
    if pixels.shape != robots.shape or pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixel_uv and robot_xy must both be Nx2 arrays.")
    if len(pixels) < 3:
        raise ValueError("At least 3 point pairs are required for affine calibration.")
    if not np.all(np.isfinite(pixels)) or not np.all(np.isfinite(robots)):
        raise ValueError("Calibration points cannot contain NaN or infinity.")
    if len(np.unique(pixels, axis=0)) != len(pixels):
        raise ValueError("Pixel calibration points contain duplicates.")

    # 在中心化、尺度归一化后检查二维覆盖，避免三个共线点也得到“零训练误差”。
    centered_pixels = pixels - pixels.mean(axis=0)
    pixel_scale = float(np.sqrt(np.mean(np.sum(centered_pixels**2, axis=1))))
    if pixel_scale < 1e-12:
        raise ValueError("Pixel calibration points have zero spatial extent.")
    normalized_design = np.column_stack(
        [
            centered_pixels / pixel_scale,
            np.ones(len(pixels), dtype=np.float64),
        ]
    )
    rank = int(np.linalg.matrix_rank(normalized_design))
    if rank < 3:
        raise ValueError(
            "Pixel calibration points are collinear; a 2D affine fit requires "
            "non-collinear coverage."
        )
    condition = float(np.linalg.cond(normalized_design))
    if not math.isfinite(condition) or condition > 1e6:
        raise ValueError(
            f"Pixel calibration layout is ill-conditioned (condition={condition:.3e})."
        )
    centered_robots = robots - robots.mean(axis=0)
    if int(np.linalg.matrix_rank(centered_robots)) < 2:
        raise ValueError(
            "Robot XY calibration points do not cover a 2D area."
        )

    # 最少 3 点可以解；实际建议 6-9 点，让误差平均掉。
    design = np.column_stack([pixels, np.ones(len(pixels), dtype=np.float64)])
    matrix_t, residuals, _, _ = np.linalg.lstsq(design, robots, rcond=None)
    transform = PixelRobotAffine(matrix_t.T)

    predicted = transform.transform_points(pixels)
    error = np.linalg.norm(predicted - robots, axis=1)
    print(f"Affine calibration mean error: {error.mean():.6f} m, max error: {error.max():.6f} m")
    if len(residuals):
        print(f"Least-squares residuals: {residuals.tolist()}")
    print(
        f"Normalized design rank: {rank}, condition number: {condition:.3e}"
    )
    return transform
def parse_args() -> argparse.Namespace:
    """定义并解析第 08 步单张图片几何推理参数。

    ``argparse.Namespace`` 可以理解为一个装参数的小对象，例如
    ``args.image`` 对应 ``--image``，``args.conf`` 对应 ``--conf``。
    """

    # description 会显示在 ``python ... --help`` 顶部。
    parser = argparse.ArgumentParser(description=__doc__)
    # 模型既可以是本地 .pt 路径，也可以是 Ultralytics 能识别的模型名称。
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "训练得到的 best.pt；省略时递归选择项目 runs 下修改时间最新的 "
            "best.pt。也可传入 Ultralytics 分割模型名。"
        ),
    )
    # 单图模式必须明确指出输入图片，不允许悄悄挑选某张图片。
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="需要分析的单张彩色图片路径。",
    )
    # 视觉配置决定四个模型类别怎样组合成紫色/黄色试管。
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "vision.yaml",
        help=(
            "试管类别配对、显示颜色和抓取点比例配置，默认 configs/vision.yaml。"
        ),
    )
    # --calib 是旧的二维像素仿射标定；它没有完整相机身份和三维外参证据。
    parser.add_argument(
        "--calib",
        type=Path,
        default=None,
        help=(
            "可选的 pixel_to_robot_affine.json 标定文件；提供后同时输出机器人 "
            "XY 和平面 yaw。"
        ),
    )
    # --eye-to-hand 是当前正式接口，只消费外部工程已经求好的标定结果。
    parser.add_argument(
        "--eye-to-hand",
        type=Path,
        default=None,
        help=(
            "外部工程导出的 eye-to-hand YAML/JSON；必须采用 "
            "p_base=T_base_from_camera@p_camera、米制单位，并包含相机内参和"
            "固定抓取平面 Z。不能与 --calib 同时使用。"
        ),
    )
    # 覆盖固定平面会使原有验证结论失效，所以后面会专门记录和限制它。
    parser.add_argument(
        "--plane-z-m",
        type=float,
        default=None,
        help=(
            "可选：覆盖 eye-to-hand 文件中的固定抓取平面 Z，单位米。"
            "只对 --eye-to-hand 生效。"
        ),
    )
    # 离线图片本身没有正在连接的相机，需要序列号证据来防止外参用错设备。
    parser.add_argument(
        "--camera-serial",
        default=None,
        help=(
            "使用 --eye-to-hand 时的 RealSense 序列号。若图片所在会话包含 "
            "session.json，会自动读取并作为身份依据；否则命令行值只是未验证"
            "声明，必须同时给出 --allow-unvalidated-eye-to-hand，且结果禁止运动。"
        ),
    )
    # 这是显式降级开关：只允许观察未验证结果，不把它包装成可执行坐标。
    parser.add_argument(
        "--allow-unvalidated-eye-to-hand",
        action="store_true",
        help=(
            "只为离线排查而允许外参状态、图片身份、内参或平面组合未能"
            "有效验证。输出会明确标记为未验证，绝不能用于机械臂运动。"
        ),
    )
    # imgsz 是模型内部缩放尺寸，不是原图最终几何坐标的尺寸。
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="YOLO 推理输入尺寸，默认 1024。",
    )
    # conf 越高，保留的低置信度实例越少。
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="保留预测实例的最低置信度，范围 0..1，默认 0.25。",
    )
    # iou 是非极大值抑制阈值，用来判断何时抑制高度重叠的较低分候选；
    # NMS 是“保留/抑制”，并不会把两个实例 mask 融合成一个。
    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="NMS 的 IoU 阈值，范围 0..1，默认 0.70。",
    )
    # device="0" 表示第一块 GPU；也可在命令行传 cpu。
    parser.add_argument("--device", default="0", help="推理设备，默认 GPU 0。")
    # 可视化文件是可选输出，不提供时只在终端打印 JSON。
    parser.add_argument(
        "--save-vis",
        type=Path,
        default=None,
        help="可选：保存分割掩膜、B/C/G 几何点和长轴的可视化图片。",
    )
    # JSON 输出便于后续脚本读取几何结果；默认同样只打印而不落盘。
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="可选：把完整几何与机器人坐标结果保存成 JSON 文件。",
    )
    # 默认拒绝覆盖是为了避免一次试验悄悄抹掉上一次结果。
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的 --output-json/--save-vis；默认拒绝覆盖。",
    )
    # 真正读取 sys.argv 并把字符串按 type=... 转成 Path、int、float 等类型。
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """兼容旧调用：严格读取视觉配置并返回普通字典。

    新代码优先使用强类型的 ``load_vision_config``；这里重新组装普通字典，
    是为了不破坏可能仍在 import 此函数的旧测试或旧脚本。
    """

    # 先走统一严格校验，避免旧接口绕过未知字段、类别顺序等规则。
    config = load_vision_config(path)
    tube_data: dict[str, dict[str, Any]] = {}
    for tube_name, spec in config.tubes.items():
        tube_data[tube_name] = {
            "body_class": spec.body_class,
            "cap_class": spec.cap_class,
            "display_color_bgr": list(spec.display_color_bgr),
        }
    return {
        "tubes": tube_data,
        "grasp": {
            "fraction_from_bottom": config.grasp_fraction_from_bottom,
        },
    }


def discover_session_camera_serial(image_path: Path) -> str | None:
    """从 ``<session>/color/<image>`` 旁的 session.json 自动读取相机序列号。"""

    # 对 .../<会话>/color/frame.jpg 连续取两个 parent，就回到 <会话>。
    session_file = image_path.expanduser().resolve().parent.parent / "session.json"
    # 老数据可能没有 session.json；这里返回 None，由上层决定是否允许降级。
    if not session_file.is_file():
        return None
    # 严格加载可以拒绝重复 JSON 键，防止 camera.serial 语义含糊。
    data = load_unique_json_mapping(session_file, description="会话元数据")
    camera = data.get("camera")
    # session.json 的 camera 必须是对象，不能是字符串或列表。
    if not isinstance(camera, dict):
        raise ValueError(f"{session_file} 的 camera 必须是 JSON 对象。")
    serial = camera.get("serial")
    # 空字符串不能作为可追溯的相机身份。
    if not isinstance(serial, str) or not serial.strip():
        raise ValueError(f"{session_file} 的 camera.serial 必须是非空字符串。")
    # 去掉用户或 JSON 中不小心写入的首尾空格。
    return serial.strip()


def discover_session_color_intrinsics(
    image_path: Path,
) -> dict[str, Any] | None:
    """读取脚本 03 保存的彩色流内参；旧会话没有该文件时返回 ``None``。

    内参中的 fx/fy/ppx/ppy 和畸变模型共同描述“像素与相机射线”的关系。
    外参即使正确，若拿错内参，像素投影到机器人平面的坐标仍会错误。
    """

    # 与 session.json 相同，intrinsics.json 放在会话根目录。
    intrinsics_file = (
        image_path.expanduser().resolve().parent.parent / "intrinsics.json"
    )
    if not intrinsics_file.is_file():
        return None
    data = load_unique_json_mapping(intrinsics_file, description="相机内参")
    # 本脚本处理彩色图片，只接受 color 流的内参。
    color = data.get("color")
    if not isinstance(color, dict):
        raise ValueError(f"{intrinsics_file} 缺少 color 内参对象。")
    return color


@dataclass(frozen=True)
class OfflineCameraIdentity:
    """单张图片可追溯到相机序列号的证据。

    ``frozen=True`` 表示对象创建后字段不可重新赋值，避免核对完成后身份信息
    被无意修改。``verified`` 只表示满足本脚本的本地追溯规则：序列号来自图片
    同一会话目录的 ``session.json``；它不是对文件真实性的密码学证明。
    ``serial_source`` 说明证据来自会话还是命令行声明。
    """

    # RealSense 的唯一设备序列号。
    serial: str
    # "session.json" 是随采集保存的证据；"command_line" 只是人工声明。
    serial_source: str
    # 只有来自同会话 session.json、且不与显式参数冲突时才为 True。
    verified: bool


def resolve_offline_camera_identity(
    image_path: Path,
    explicit_camera_serial: str | None,
    *,
    allow_unvalidated: bool,
) -> OfflineCameraIdentity:
    """确定离线图片相机身份；命令行序列号不能充当采集证据。

    ``session.json`` 是脚本 03 在采集时写下的随图证据，因此可把身份标记为已
    核对。命令行字符串只是操作者声明，只允许在显式未验证模式下用于排查。
    """

    # 先设为 None；只有收到非空字符串时才保存命令行序列号。
    explicit = None
    if isinstance(explicit_camera_serial, str) and explicit_camera_serial.strip():
        explicit = explicit_camera_serial.strip()
    # 优先寻找与图片一起保存的采集证据。
    session_serial = discover_session_camera_serial(image_path)
    if session_serial is not None:
        # 用户同时填写序列号时，它必须与采集记录完全一致。
        if explicit is not None and explicit != session_serial:
            raise ValueError(
                "--camera-serial 与图片会话 session.json 记录不一致："
                f"命令行={explicit!r}，会话={session_serial!r}。"
            )
        # 按本项目本地追溯规则，同会话 session.json 可标为 verified；这表示
        # 元数据来源合格，不代表脚本重新连接了相机或独立鉴定了文件真伪。
        return OfflineCameraIdentity(
            serial=session_serial,
            serial_source="session.json",
            verified=True,
        )

    # 无会话证据且连人工声明也没有，就无法知道该使用哪台相机的外参。
    if explicit is None:
        raise ValueError(
            "使用 --eye-to-hand 时无法确认图片来自哪台相机："
            "图片会话缺少 session.json；离线排查时还需同时提供 "
            "--camera-serial 和 --allow-unvalidated-eye-to-hand。"
        )
    # 即使给了 --camera-serial，它仍只是事后输入，默认不能冒充采集证据。
    if not allow_unvalidated:
        raise ValueError(
            "图片会话缺少 session.json；命令行 --camera-serial 只是操作者声明，"
            "不能验证图片相机身份。若仅做离线排查，必须显式添加 "
            "--allow-unvalidated-eye-to-hand；结果不得用于机械臂。"
        )
    # 只有显式开启未验证模式才返回该身份，并明确把 verified 保持为 False。
    return OfflineCameraIdentity(
        serial=explicit,
        serial_source="command_line",
        verified=False,
    )


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """把模型掩膜恢复到原图尺寸，并转换为布尔前景掩膜。"""

    # NumPy 二维数组的 shape 顺序是 (高, 宽)，与 OpenCV Size 的 (宽, 高)
    # 正好相反，这是图像处理中常见的初学者易错点。
    if mask.shape == (height, width):
        return mask.astype(bool)
    # 最近邻插值不会在 0/1 掩膜边界引入新的灰度类别。
    resized = cv2.resize(
        mask.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    # 模型掩膜通常是 0..1 浮点值，大于 0.5 的像素视作前景 True。
    return resized > 0.5


@dataclass(frozen=True)
class MaskDetection:
    """一个类别中置信度最高的分割实例。

    这里不用普通三元素 tuple，是为了让调用处通过 ``.mask``、
    ``.confidence`` 和 ``.class_id`` 清楚表达每个值的意义。
    """

    # 与原图同尺寸的布尔掩膜。
    mask: np.ndarray
    # YOLO 为这个实例给出的置信度。
    confidence: float
    # 模型内部的数值类别 ID；类别语义仍以 result.names 为准。
    class_id: int


def best_detection_by_name(
    result: Any,
    class_name: str,
    image_shape: tuple[int, int],
) -> MaskDetection | None:
    """返回指定类别中置信度最高的掩膜及其置信度。"""

    # 分割结果缺少 masks 或 boxes 时，没有可用实例。
    if result.masks is None or result.boxes is None:
        return None

    # 同一类别可能预测多个实例；当前实验每种颜色只有一支试管，所以只取
    # 置信度最高者。若以后同色试管超过一支，需要改为返回实例列表并做配对。
    # image_shape 按 NumPy 习惯传入 (height, width)。
    height, width = image_shape
    # detach() 脱离 PyTorch 梯度图，cpu() 搬到主存，numpy() 转成 NumPy。
    # 推理阶段不需要梯度；转换后便于用普通 Python/NumPy 处理。
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confs = result.boxes.conf.detach().cpu().numpy()
    masks = result.masks.data.detach().cpu().numpy()
    names = result.names

    # best_index 记录当前最高分实例在批量结果中的下标。
    best_index = None
    best_conf = -1.0
    # 用列表下标 index 读取该位置的类别 ID。
    for index in range(len(classes)):
        cls_id = classes[index]
        # result.names 把数值类别 ID 映射回 p-body、p-cap 等字符串。
        if names[int(cls_id)] == class_name and confs[index] > best_conf:
            best_index = index
            best_conf = float(confs[index])

    # 遍历结束仍为 None，说明目标类别没有任何检测。
    if best_index is None:
        return None
    # 输出前把掩膜恢复到原图尺寸，保证所有几何点使用原图像素坐标。
    return MaskDetection(
        mask=resize_mask(masks[best_index], width, height),
        confidence=float(confs[best_index]),
        class_id=int(classes[best_index]),
    )


def detection_count_by_name(result: Any, class_name: str) -> int:
    """统计某类别实例数，用于拒绝尚未实现空间配对的多目标画面。"""

    if result.boxes is None:
        return 0
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    # 每匹配一个目标类别就把计数加一。
    count = 0
    for class_id in classes:
        if str(result.names[int(class_id)]) == class_name:
            count += 1
    return count


def find_latest_best_weight() -> Path:
    """递归寻找项目 ``runs`` 下修改时间最新的训练最佳权重。

    递归搜索同时兼容 ``runs/segment/<name>/weights/best.pt`` 和早期
    Ultralytics 产生的 ``runs/segment/runs/segment/...`` 嵌套路径。
    """

    # rglob 会递归进入所有实验子目录；只保留真实文件。
    candidates: list[Path] = []
    for path in (ROOT / "runs").rglob("best.pt"):
        if path.is_file():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("runs 下没有 best.pt，请通过 --model 指定模型。")
    # 逐个比较修改时间。这里选“最新写入”，不代表它的指标一定最高。
    latest_path = candidates[0]
    latest_time = latest_path.stat().st_mtime
    for path in candidates[1:]:
        modified_time = path.stat().st_mtime
        if modified_time > latest_time:
            latest_path = path
            latest_time = modified_time
    return latest_path.resolve()


def resolve_model_argument(value: str | None) -> str:
    """解析本地权重路径，并保留对 Ultralytics 官方模型名称的支持。"""

    # 用户省略 --model 时才自动查找本项目的最新 best.pt。
    if value is None:
        return str(find_latest_best_weight())
    # expanduser 把 ~/weights.pt 展开成完整主目录路径。
    candidate = Path(value).expanduser()
    # 存在的本地文件规范化为绝对路径；不存在时保留原字符串，因为它可能是
    # Ultralytics 可下载/识别的官方模型名称。
    if candidate.is_file():
        return str(candidate.resolve())
    return value


def tube_specs_from_config(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """读取并校验 ``vision.yaml`` 中紫色、黄色试管的类别配对配置。"""

    # parse_vision_config 负责检查必需字段、未知字段、类别重复和数值范围。
    parsed = parse_vision_config(config)
    # 返回旧版代码熟悉的 dict 结构；元组颜色仍按 OpenCV 的 BGR 顺序。
    tube_specs: dict[str, dict[str, Any]] = {}
    for tube_name, spec in parsed.tubes.items():
        tube_specs[tube_name] = {
            "body_class": spec.body_class,
            "cap_class": spec.cap_class,
            "display_color_bgr": spec.display_color_bgr,
        }
    return tube_specs


def draw_labeled_tube_pose(
    image: np.ndarray,
    tube_name: str,
    pose: TubePose2D,
    color_bgr: tuple[int, int, int],
    body_confidence: float,
    cap_confidence: float | None,
) -> np.ndarray:
    """绘制长轴和抓取点；只有检测到管盖时才把两端称为 B/C。"""

    # copy 避免调用者传进来的原图被原地修改。
    output = image.copy()
    # 几何库保存浮点坐标；OpenCV 绘图函数需要整数像素坐标。
    bottom = tuple(np.round(pose.bottom_xy).astype(int))
    cap = tuple(np.round(pose.cap_xy).astype(int))
    grasp = tuple(np.round(pose.grasp_xy).astype(int))
    center = tuple(np.round(pose.center_xy).astype(int))

    # 长轴连接两个管身端点；抓取点 G 使用醒目的黄色。
    cv2.line(output, bottom, cap, color_bgr, 3, cv2.LINE_AA)
    cv2.circle(output, grasp, 7, (0, 255, 255), -1)

    if cap_confidence is not None:
        # B=bottom（管底）、C=cap-side（靠盖一端）。C 是“管身长轴端点中
        # 更靠近管盖的那个点”，不是管盖 mask 的中心；管盖只负责判断头尾。
        endpoint_a_label = "B"
        endpoint_b_label = "C"
        endpoint_a_color = (255, 0, 0)
        endpoint_b_color = (0, 0, 255)
    else:
        # 漏检管盖时 PCA 长轴没有方向，不能把任意端点误称为管底或盖子端。
        endpoint_a_label = "E1"
        endpoint_b_label = "E2"
        endpoint_a_color = (180, 180, 180)
        endpoint_b_color = (180, 180, 180)

    cv2.circle(output, bottom, 6, endpoint_a_color, -1)
    cv2.circle(output, cap, 6, endpoint_b_color, -1)
    cv2.putText(
        output,
        endpoint_a_label,
        (bottom[0] + 7, bottom[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        endpoint_a_color,
        2,
    )
    cv2.putText(
        output,
        endpoint_b_label,
        (cap[0] + 7, cap[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        endpoint_b_color,
        2,
    )
    # G=grasp（抓取点）；无管盖时强制使用管身中点。
    cv2.putText(
        output,
        "G",
        (grasp[0] + 7, grasp[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    # f"{value:.2f}" 把置信度格式化为两位小数。
    cap_text = "missing"
    if cap_confidence is not None:
        cap_text = f"{cap_confidence:.2f}"
    label = f"{tube_name} body={body_confidence:.2f} cap={cap_text}"
    cv2.putText(
        output,
        label,
        (center[0] + 10, center[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color_bgr,
        2,
        cv2.LINE_AA,
    )
    return output


def tube_pose_to_output(
    pose: TubePose2D,
    body_class: str,
    cap_class: str,
    body_confidence: float,
    cap_confidence: float | None,
) -> dict[str, Any]:
    """把 ``TubePose2D`` 和置信度整理成可直接 JSON 序列化的字典。

    所有 ``*_xy`` 图像坐标均采用 OpenCV 约定：原点在左上角，x 向右、y 向下。
    """

    # 公共字段不依赖“长轴方向是否已由管盖确认”。
    output: dict[str, Any] = {
        "detected": True,
        "body_class": body_class,
        "cap_class": cap_class,
        "body_confidence": body_confidence,
        "cap_confidence": cap_confidence,
        "direction_resolved_by_cap": cap_confidence is not None,
        "center_xy": pose.center_xy,
        "grasp_xy": pose.grasp_xy,
        # 供离线候选报告做保守质量门；由 tube_pose_from_masks 产生的姿态始终
        # 包含这两个已经过有限值和下限检查的指标。其中 pca_aspect_ratio 是
        # PCA/SVD 两个奇异值的比值，不是外接框长宽比。
        "length_px": pose.length_px,
        "pca_aspect_ratio": pose.pca_aspect_ratio,
    }
    if cap_confidence is not None:
        # 管盖位于 C 端，因此 axis/angle 是有方向的 B→C。
        output.update(
            {
                "bottom_xy": pose.bottom_xy,
                "cap_xy": pose.cap_xy,
                "axis_bottom_to_cap_xy": pose.axis_xy,
                "angle_rad_image": pose.angle_rad,
                "angle_deg_image": math.degrees(pose.angle_rad),
            }
        )
    else:
        # 无管盖时两个端点及 PCA 轴都没有方向；模 180° 后才是稳定的轴角。
        output.update(
            {
                "endpoint1_xy": pose.bottom_xy,
                "endpoint2_xy": pose.cap_xy,
                "axis_angle_deg_mod_180": math.degrees(pose.angle_rad) % 180.0,
            }
        )
    return output


def write_text_atomic(path: Path, text: str, *, overwrite: bool) -> None:
    """暂存文本后原子发布；默认用硬链接保证绝不覆盖已有文件。

    “原子发布”表示其他进程只会看到完整旧文件或完整新文件，不会看到写到
    一半的 JSON。默认模式的 ``os.link`` 还会在目标已存在时由操作系统报错。
    """

    # parents=True 会补齐多级父目录；exist_ok=True 允许目录已经存在。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 先设为 None，确保创建临时文件中途失败时 finally 仍可安全判断。
    temporary_path: Path | None = None
    try:
        # 临时文件放在目标同一目录，以确保最终替换位于同一文件系统。
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            # flush 把 Python 缓冲区交给系统；fsync 再要求系统同步到存储设备。
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            # os.replace 会原子替换已有目标，仅在用户明确 --overwrite 时使用。
            os.replace(temporary_path, path)
        else:
            # 临时文件与目标在同一目录，link 的“目标必须不存在”由内核原子保证，
            # 消除 exists() 检查与最终发布之间的 TOCTOU 覆盖窗口。
            os.link(temporary_path, path)
    finally:
        # 发布成功或失败都清理临时名字；missing_ok 避免已被 replace 后报错。
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_image_atomic(
    path: Path,
    image: np.ndarray,
    *,
    overwrite: bool,
) -> None:
    """暂存 OpenCV 图像后原子发布，并在默认模式中保证不覆盖。

    图片编码也可能失败或进程中断，所以不能直接向最终文件写入。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    # 若用户没有提供扩展名，临时文件使用 .png，方便 OpenCV 选择编码器。
    suffix = path.suffix
    if not suffix:
        suffix = ".png"
    # mkstemp 返回一个已创建文件的底层描述符和路径，避免临时文件名竞争。
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=suffix,
    )
    # cv2.imwrite 会自己打开路径，先关闭 mkstemp 返回的描述符。
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        # OpenCV 保存失败时返回 False，而不一定抛出异常，因此必须主动检查。
        if not cv2.imwrite(str(temporary_path), image):
            raise RuntimeError(f"可视化保存失败：{path}")
        if overwrite:
            os.replace(temporary_path, path)
        else:
            os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def new_image_geometry_output(
    *,
    image_path: Path,
    model_argument: str,
    config_path: Path,
    width: int,
    height: int,
    grasp_fraction: float,
) -> dict[str, Any]:
    """建立带固定版本/类型标识的单图几何报告。

    ``schema_version`` 和 ``type`` 让下游脚本先确认“读到的是什么版本、什么
    类型”，再访问 tubes 字段，避免把任意 JSON 当作抓取候选。
    """

    # tubes 初始为空，后续循环会为 purple、yellow 分别填入成功或失败结果。
    return {
        "schema_version": 1,
        "type": "image_geometry",
        "image": str(image_path),
        "model": model_argument,
        "config": str(config_path),
        "image_size": {"width": width, "height": height},
        "grasp_fraction_from_bottom": grasp_fraction,
        "tubes": {},
    }


def main() -> int:
    """执行单图分割、类别配对、几何计算、JSON 输出和可选可视化。"""

    # 第一步先解析并校验所有纯参数；尽量在加载模型前快速暴露输入错误。
    args = parse_args()
    # 旧二维仿射和新 eye-to-hand 是两套不同坐标模型，同时提供会语义冲突。
    if args.calib is not None and args.eye_to_hand is not None:
        raise ValueError("--calib 与 --eye-to-hand 不能同时使用。")
    # 平面高度属于 eye-to-hand 射线求交参数，单独使用没有意义。
    if args.plane_z_m is not None and args.eye_to_hand is None:
        raise ValueError("--plane-z-m 只能与 --eye-to-hand 一起使用。")
    # “允许未验证”只能降低 eye-to-hand 显示门槛，不能成为孤立开关。
    if (
        args.allow_unvalidated_eye_to_hand
        and args.eye_to_hand is None
    ):
        raise ValueError(
            "--allow-unvalidated-eye-to-hand 只能与 --eye-to-hand 一起使用。"
        )
    if args.imgsz <= 0:
        raise ValueError("--imgsz 必须大于 0。")
    # isfinite 同时排除 NaN 和正负无穷；普通范围比较不能可靠表达这些坏值。
    if (
        not math.isfinite(args.conf)
        or not math.isfinite(args.iou)
        or not 0.0 <= args.conf <= 1.0
        or not 0.0 <= args.iou <= 1.0
    ):
        raise ValueError("--conf 和 --iou 必须是位于 0..1 的有限数值。")
    if args.plane_z_m is not None and not math.isfinite(args.plane_z_m):
        raise ValueError("--plane-z-m 必须是有限数值。")
    # 输入图、JSON 输出和可视化输出不能是同一个路径，否则可能覆盖原图，
    # 或让两种输出彼此覆盖。resolve 后比较可识别 ./a 与 /abs/a 是同一路径。
    protected_paths = [args.image.expanduser().resolve()]
    for path in (args.output_json, args.save_vis):
        if path is not None:
            protected_paths.append(path.expanduser().resolve())
    if len(protected_paths) != len(set(protected_paths)):
        raise ValueError(
            "--image、--output-json 和 --save-vis 必须指向互不相同的文件。"
        )
    # 启动推理前先做一次覆盖检查，避免昂贵计算完成后才发现目标已存在。
    if not args.overwrite:
        existing_outputs: list[Path] = []
        for path in (args.output_json, args.save_vis):
            if path is not None:
                resolved_path = path.expanduser().resolve()
                if resolved_path.exists():
                    existing_outputs.append(resolved_path)
        if existing_outputs:
            raise FileExistsError(
                "输出文件已存在，默认拒绝覆盖；请更换路径或显式添加 "
                f"--overwrite：{existing_outputs}"
            )

    # 放在 main 里面导入，方便没有 YOLO 时先运行 --help。
    from ultralytics import YOLO

    # vision.yaml 把 p-body/p-cap 配成紫色试管，把 y-body/y-cap 配成黄色
    # 试管，并给出从管底到盖子方向的抓取比例。
    vision_config = load_vision_config(args.config)
    # 将不可变配置对象整理成循环中便于按字符串键读取的字典。
    tube_specs: dict[str, dict[str, Any]] = {}
    for tube_name, spec in vision_config.tubes.items():
        tube_specs[tube_name] = {
            "body_class": spec.body_class,
            "cap_class": spec.cap_class,
            "display_color_bgr": spec.display_color_bgr,
        }
    # 例如 0.5 表示 G 位于 B→C 长轴的中点。
    grasp_fraction = vision_config.grasp_fraction_from_bottom
    # 不传 --model 时自动选最新的 best.pt；显式传入时可固定到某次实验。
    model_argument = resolve_model_argument(args.model)
    # YOLO(...) 加载权重和模型元数据，但此时尚未对图片推理。
    model = YOLO(model_argument)
    # 嵌套推导式按 vision.yaml 的试管顺序展开为：
    # (p-body, p-cap, y-body, y-cap)。
    configured_class_names: list[str] = []
    for spec in tube_specs.values():
        configured_class_names.append(spec["body_class"])
        configured_class_names.append(spec["cap_class"])
    configured_class_order = tuple(configured_class_names)
    # 必须在读取图片和 model.predict() 之前失败，避免自动选到其他实验权重后
    # 仍做一次昂贵且语义错误的推理。
    validate_tube_model_contract(
        task=model.task,
        names=model.names,
        configured_class_order=configured_class_order,
    )

    # 统一成绝对路径，让报告在不同工作目录中仍能追溯输入文件。
    image_path = args.image.expanduser().resolve()
    # cv2.imread 默认返回 BGR、形状为 (height, width, 3) 的 NumPy 数组。
    image = cv2.imread(str(image_path))
    # OpenCV 读取不存在、损坏或不支持的图片通常返回 None。
    if image is None:
        raise FileNotFoundError(image_path)
    # shape 前两维依次是高和宽。
    height, width = image.shape[:2]

    # retina_masks=True 请求原图分辨率实例掩膜，有利于像素几何；若某版本返回
    # 的 mask 尺寸仍不同，best_detection_by_name 中的 resize_mask 会兜底。
    # predict 返回结果列表；这里只有一张输入图片，所以取第 0 个结果。
    result = model.predict(
        source=image,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        # 项目视觉脚本统一固定使用 FP32。
        half=False,
        retina_masks=True,
        verbose=False,
    )[0]

    # 先建立报告“外壳”，后续每种颜色都填入 tubes。
    output = new_image_geometry_output(
        image_path=image_path,
        model_argument=model_argument,
        config_path=args.config.expanduser().resolve(),
        width=width,
        height=height,
        grasp_fraction=grasp_fraction,
    )
    # 先显示模型的分割掩膜、类别和置信度，再叠加几何点和长轴。
    vis = result.plot(conf=True, labels=True, boxes=True, masks=True, color_mode="class")
    # 只有用户提供对应参数时才加载标定；否则值保持 None。
    pixel_calibration = None
    if args.calib:
        pixel_calibration = PixelRobotAffine.load(args.calib)
    eye_to_hand = None
    if args.eye_to_hand:
        eye_to_hand = ExternalEyeToHandCalibration.load(args.eye_to_hand)
    # 以下状态既用于安全判断，也会写入 JSON，方便下游知道坐标是否可信。
    camera_serial: str | None = None
    camera_serial_source: str | None = None
    camera_identity_verified = False
    eye_to_hand_effectively_validated = False
    plane_override_changed = False
    stream_intrinsics_verified = False
    if eye_to_hand is not None:
        # 本脚本的像素来自彩色图；深度光学坐标系外参不能直接套用。
        if eye_to_hand.camera_frame != "color_optical":
            raise ValueError(
                "--image 是彩色分割图，但 eye-to-hand 的 camera.frame 为 "
                f"{eye_to_hand.camera_frame!r}；必须提供 color_optical 外参。"
            )
        # math.isclose 用 1e-9 m 绝对误差比较，避免浮点表示噪声误判为覆盖。
        plane_override_changed = (
            args.plane_z_m is not None
            and not math.isclose(
                float(args.plane_z_m),
                eye_to_hand.plane_z_m,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
        # “文件验证通过”在这里准确地说，是外部文件的 validation 元数据声明
        # validated 且结构检查通过；本脚本不重新求解或复现实验验证。平面没有
        # 被改动时才暂时保留该状态，后面还要叠加相机身份和内参两项条件。
        eye_to_hand_effectively_validated = (
            eye_to_hand.validation.is_validated and not plane_override_changed
        )
        # 默认 fail closed：验证不完整就终止，而不是悄悄给出看似可用的坐标。
        # allow_unvalidated 只允许“验证证据缺失/平面覆盖”降级显示，不会放过
        # 已知的序列号、分辨率、坐标系或内参不匹配。
        if (
            not eye_to_hand_effectively_validated
            and not args.allow_unvalidated_eye_to_hand
        ):
            # 若命令行覆盖了平面高度，就换成更具体的报错原因。
            reason = "validation.status 不是 validated"
            if plane_override_changed:
                reason = "命令行覆盖的 plane_z_m 没有包含在外部独立验证中"
            raise ValueError(
                f"eye-to-hand 当前不能视为已验证：{reason}。"
                "若只是离线排查，可显式添加 "
                "--allow-unvalidated-eye-to-hand；机械臂流程不得使用。"
            )
        # 核对图片会话记录与命令行序列号，防止把 A 相机外参套到 B 相机图片。
        identity = resolve_offline_camera_identity(
            args.image,
            args.camera_serial,
            allow_unvalidated=args.allow_unvalidated_eye_to_hand,
        )
        camera_serial = identity.serial
        camera_serial_source = identity.serial_source
        camera_identity_verified = identity.verified
        # 分辨率和 camera_frame 也必须与外参文件声明完全相符。
        eye_to_hand.validate_stream(
            serial=camera_serial,
            width=width,
            height=height,
            camera_frame="color_optical",
        )
        # 读取采集时实际保存的彩色内参，并在小容差内核对焦距、主点、
        # 畸变模型和有效畸变系数；“none + 若干零系数”按相同语义处理。
        saved_intrinsics = discover_session_color_intrinsics(args.image)
        if saved_intrinsics is not None:
            eye_to_hand.validate_intrinsics(
                fx=saved_intrinsics.get("fx"),
                fy=saved_intrinsics.get("fy"),
                ppx=saved_intrinsics.get("ppx"),
                ppy=saved_intrinsics.get("ppy"),
                distortion_model=saved_intrinsics.get(
                    "distortion_model",
                    saved_intrinsics.get("model"),
                ),
                coeffs=saved_intrinsics.get("coeffs", []),
            )
            # validate_intrinsics 没有抛出异常，才能把该证据标成 True。
            stream_intrinsics_verified = True
        elif not args.allow_unvalidated_eye_to_hand:
            raise ValueError(
                "图片会话缺少 intrinsics.json，无法确认实际彩色流内参与"
                " eye-to-hand 文件一致。若仅做离线排查，可显式添加 "
                "--allow-unvalidated-eye-to-hand；坐标不得用于机械臂。"
            )
        # 三个条件必须同时为真：外部验证元数据/平面组合仍适用、
        # 本地图片身份证据合格、会话内参与标定文件匹配。
        eye_to_hand_effectively_validated = (
            eye_to_hand_effectively_validated
            and camera_identity_verified
            and stream_intrinsics_verified
        )

    # JSON 中明确记录使用了哪套标定，禁止下游靠字段形状猜测。
    if pixel_calibration is not None:
        output["calibration"] = {
            "type": "pixel_to_robot_affine",
            "file": str(args.calib.expanduser().resolve()),
            "validated": False,
            "warning": "旧二维仿射格式不含相机身份和独立验证元数据。",
        }
    elif eye_to_hand is not None:
        # T_base_from_camera 的含义是把相机坐标齐次点左乘到基座坐标，
        # 绝不能反过来使用。
        selected_plane_z = eye_to_hand.plane_z_m
        if args.plane_z_m is not None:
            selected_plane_z = float(args.plane_z_m)
        output["calibration"] = {
            "type": "external_eye_to_hand",
            "file": str(args.eye_to_hand.expanduser().resolve()),
            "transform_convention": "p_base = T_base_from_camera @ p_camera",
            "camera_serial": camera_serial,
            "camera_serial_source": camera_serial_source,
            "camera_identity_verified": camera_identity_verified,
            "camera_frame": eye_to_hand.camera_frame,
            "stream_intrinsics_verified": stream_intrinsics_verified,
            "plane_z_m": selected_plane_z,
            "plane_z_overridden": plane_override_changed,
            "effective_validated": eye_to_hand_effectively_validated,
            "validation": eye_to_hand.validation.as_dict(),
        }
        # 降级模式下把所有不可信原因累积起来，而不是只报告第一项。
        warnings: list[str] = []
        if plane_override_changed:
            warnings.append(
                "plane_z_m 被命令行覆盖；该平面没有包含在原验证中，"
                "机器人坐标仅供离线排查。"
            )
        if not camera_identity_verified:
            warnings.append(
                "camera_serial 仅来自命令行，不是采集会话证据；"
                "相机身份未验证，坐标不得用于机械臂。"
            )
        if not stream_intrinsics_verified:
            warnings.append(
                "旧图片会话缺少 intrinsics.json，未能独立核对实际流内参；"
                "坐标可离线查看，但不得直接进入动作计划。"
            )
        if warnings:
            # join 用空格把多条中文警告合成一个 JSON 字符串。
            output["calibration"]["warning"] = " ".join(warnings)

    # 依次处理 purple 和 yellow；每组由一个 body 类与一个 cap 类构成。
    for tube_name, spec in tube_specs.items():
        # 取出当前颜色对应的模型类别名。
        body_class = spec["body_class"]
        cap_class = spec["cap_class"]
        # 先数实例而不是马上只取最高分，防止同色两支试管被错误拼接。
        body_count = detection_count_by_name(result, body_class)
        cap_count = detection_count_by_name(result, cap_class)
        if body_count > 1 or cap_count > 1:
            # 当前版本没有 body/cap 空间匹配算法，因此歧义场景必须拒绝几何。
            output["tubes"][tube_name] = {
                "detected": False,
                "body_class": body_class,
                "cap_class": cap_class,
                "body_instances": body_count,
                "cap_instances": cap_count,
                "reason": (
                    "同色多实例的 body/cap 空间配对尚未实现，"
                    "为避免错误抓取已拒绝输出几何。"
                ),
            }
            # continue 直接开始下一种颜色，不执行本轮后续计算。
            continue
        # 唯一实例场景下分别取得最高置信度管身和管盖掩膜。
        body_detection = best_detection_by_name(result, body_class, (height, width))
        cap_detection = best_detection_by_name(result, cap_class, (height, width))
        cap_confidence = None
        cap_mask = None
        selected_grasp_fraction = 0.5
        if cap_detection is not None:
            cap_confidence = cap_detection.confidence
            cap_mask = cap_detection.mask
            selected_grasp_fraction = grasp_fraction

        if body_detection is None:
            # 没有管身就无法做 PCA；即使单独检测到管盖也不能生成试管姿态。
            output["tubes"][tube_name] = {
                "detected": False,
                "body_class": body_class,
                "cap_class": cap_class,
                "reason": f"没有检测到 {body_class}",
                "cap_confidence": cap_confidence,
            }
            continue

        # 几何库先保留管身最大连通域，按初次 PCA 投影去掉两端各 5% 的离群
        # 像素，再重新拟合长轴和端点。盖子 mask 的中心只用来判断哪个管身
        # 端点更靠近盖子，并且还要通过“靠近长轴、靠近某一端”的配对检查。
        # 漏检管盖时无法判断从哪端量抓取比例，因此只使用无方向中点 0.5。
        try:
            # tube_pose_from_masks 内部会做连通域、PCA 长轴长度/长宽比等检查。
            tube_pose = tube_pose_from_masks(
                body_detection.mask,
                cap_mask,
                selected_grasp_fraction,
            )
        except ValueError as exc:
            # 单种颜色几何失败不应阻止另一颜色被分析，所以记录原因后继续。
            output["tubes"][tube_name] = {
                "detected": False,
                "body_class": body_class,
                "cap_class": cap_class,
                "body_confidence": body_detection.confidence,
                "cap_confidence": cap_confidence,
                "reason": f"几何质量检查失败：{exc}",
            }
            continue
        # 把 NumPy/数据类形式的姿态整理为统一输出字段。
        tube_output = tube_pose_to_output(
            tube_pose,
            body_class,
            cap_class,
            body_detection.confidence,
            cap_confidence,
        )

        # 有标定文件时始终可以转换中点 G；只有管盖存在、方向可靠时，才输出
        # 机器人坐标中的 B/C 和 yaw，避免把 PCA 任意符号当成抓取朝向。
        if pixel_calibration is not None:
            # 旧二维仿射直接把像素 [u,v] 映射为工作平面 [x,y]（米）。
            grasp_robot = pixel_calibration.transform_point(tube_pose.grasp_xy)
            robot_output: dict[str, Any] = {
                "grasp_xy_m": grasp_robot,
                "direction_resolved_by_cap": cap_detection is not None,
                "source": "pixel_to_robot_affine",
            }
            if cap_detection is not None:
                # B、C 分别转换后，通过 atan2(Δy, Δx) 得到基座平面 yaw。
                bottom_robot = pixel_calibration.transform_point(
                    tube_pose.bottom_xy
                )
                cap_robot = pixel_calibration.transform_point(tube_pose.cap_xy)
                yaw_robot = math.atan2(
                    cap_robot[1] - bottom_robot[1],
                    cap_robot[0] - bottom_robot[0],
                )
                robot_output.update(
                    {
                        "bottom_xy_m": bottom_robot,
                        "cap_xy_m": cap_robot,
                        "yaw_rad": yaw_robot,
                        "yaw_deg": math.degrees(yaw_robot),
                    }
                )
            tube_output["robot"] = robot_output
        elif eye_to_hand is not None:
            # 新流程先按内参/畸变模型把 G 像素反投影成相机射线，再用外参
            # 旋转射线、确定基座中的相机光心，并与 z=plane_z_m 平面求交。
            # 这里没有读取深度；它明确假设 B/C/G 都落在同一个固定基座 Z 平面。
            grasp_robot_xyz = eye_to_hand.pixel_to_base_plane(
                tube_pose.grasp_xy,
                plane_z_m=args.plane_z_m,
            )
            robot_output = {
                "grasp_xyz_m": grasp_robot_xyz,
                "direction_resolved_by_cap": cap_detection is not None,
                "source": "external_eye_to_hand_ray_plane",
                "calibration_validated": eye_to_hand_effectively_validated,
            }
            if cap_detection is not None:
                # bcg_to_base 对 B/C/G 使用同一套相机模型、外参和固定平面。
                base_points = eye_to_hand.bcg_to_base(
                    tube_pose.bottom_xy,
                    tube_pose.cap_xy,
                    tube_pose.grasp_xy,
                    plane_z_m=args.plane_z_m,
                )
                bottom_robot_xyz = base_points["B"]
                cap_robot_xyz = base_points["C"]
                # 使用基座坐标的 B→C 平面向量计算 yaw，而非直接沿用图像角度。
                yaw_robot = math.atan2(
                    cap_robot_xyz[1] - bottom_robot_xyz[1],
                    cap_robot_xyz[0] - bottom_robot_xyz[0],
                )
                robot_output.update(
                    {
                        "bottom_xyz_m": bottom_robot_xyz,
                        "cap_xyz_m": cap_robot_xyz,
                        "yaw_rad": yaw_robot,
                        "yaw_deg": math.degrees(yaw_robot),
                    }
                )
            tube_output["robot"] = robot_output

        # 保存当前颜色结果，并在可视化上继续叠加；下轮会处理另一种颜色。
        output["tubes"][tube_name] = tube_output
        vis = draw_labeled_tube_pose(
            vis,
            tube_name,
            tube_pose,
            spec["display_color_bgr"],
            body_detection.confidence,
            cap_confidence,
        )

    # ensure_ascii=False 保留中文；allow_nan=False 禁止产生非标准 JSON 的 NaN。
    output_json = json.dumps(
        output,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    # 无论是否指定 --output-json，终端都打印完整结果，便于立即检查。
    print(output_json)

    if args.output_json:
        # 输出前再次检查文件是否在推理期间被其他进程创建，缩小竞争窗口。
        json_path = args.output_json.expanduser().resolve()
        if json_path.exists() and not args.overwrite:
            raise FileExistsError(
                "--output-json 在运行期间已出现，默认拒绝覆盖；"
                f"请更换路径或显式添加 --overwrite：{json_path}"
            )
        # 文件末尾加换行，便于终端工具和文本编辑器处理。
        write_text_atomic(
            json_path,
            output_json + "\n",
            overwrite=args.overwrite,
        )
        print(f"saved JSON: {json_path}")

    if args.save_vis:
        # 可视化与 JSON 使用相同的“默认不覆盖 + 原子发布”策略。
        save_path = args.save_vis.expanduser().resolve()
        if save_path.exists() and not args.overwrite:
            raise FileExistsError(
                "--save-vis 在运行期间已出现，默认拒绝覆盖；"
                f"请更换路径或显式添加 --overwrite：{save_path}"
            )
        write_image_atomic(save_path, vis, overwrite=args.overwrite)
        print(f"saved visualization: {save_path}")
    # 进程退出码 0 表示所有请求的输出都已成功完成。
    return 0


# 只有直接运行脚本时才进入 main；import 本文件做测试不会自动推理。
if __name__ == "__main__":
    raise SystemExit(main())
