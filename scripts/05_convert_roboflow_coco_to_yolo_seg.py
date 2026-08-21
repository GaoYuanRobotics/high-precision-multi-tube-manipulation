#!/usr/bin/env python3
"""第 05 步：把 Roboflow COCO Segmentation 转成 YOLO Seg 数据集。

本文件直接负责数据校验、类别映射、原图分组划分和文件生成。

Roboflow 会为同一张原图生成多个增强版本。为了避免训练集和验证集之间
出现同源图片泄漏，本模块使用 COCO ``images[].extra.name`` 作为分组键，
同一原图的所有增强版本一定进入同一个 split。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_class_map(names: Sequence[str]) -> dict[str, int]:
    """校验类别名称，并建立“名称 -> YOLO 类别编号”映射。"""

    if not names:
        raise ValueError("至少需要一个类别名称。")
    normalized = []
    class_map = {}
    class_id = 0
    for name in names:
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("类别名称不能为空。")
        if clean_name in class_map:
            raise ValueError("类别名称不能重复。")
        normalized.append(clean_name)
        class_map[clean_name] = class_id
        class_id += 1
    return class_map


def write_dataset_yaml(
    dataset_dir: Path,
    class_names: Sequence[str],
    yaml_path: Path | None = None,
) -> Path:
    """写出 Ultralytics 训练所需的 tube_seg.yaml。"""

    if yaml_path is None:
        yaml_path = dataset_dir / "tube_seg.yaml"

    names_by_id = {}
    class_id = 0
    for name in class_names:
        names_by_id[class_id] = str(name)
        class_id += 1

    content = yaml.safe_dump(
        {
            "path": str(dataset_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(class_names),
            "names": names_by_id,
        },
        allow_unicode=True,
        sort_keys=False,
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def class_names_from_file(path: Path) -> list[str]:
    """读取 classes.txt 中的有效类别行。"""

    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        clean_line = line.strip()
        if clean_line and not clean_line.startswith("#"):
            names.append(clean_line)
    if not names:
        raise ValueError(f"类别文件中没有有效类别：{path}")
    return names


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """拒绝任意层级的 JSON 重复键，避免后一个 COCO 字段静默覆盖前一个。"""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"COCO JSON 中存在重复键 {key!r}，输入已拒绝。")
        result[key] = value
    return result


@dataclass(frozen=True)
class RoboflowConversionStats:
    """一次 Roboflow COCO 转换的汇总结果。"""

    images: int
    annotations: int
    labels_written: int
    source_groups: int
    train_groups: int
    val_groups: int
    train_images: int
    val_images: int
    train_labels: int
    val_labels: int


@dataclass(frozen=True)
class _PreparedImage:
    """完成校验、等待写入 YOLO 数据集的一张图片。"""

    image_id: int
    source_path: Path
    output_name: str
    source_group: str
    label_lines: tuple[str, ...]


def find_coco_annotation(input_path: Path) -> tuple[Path, Path]:
    """定位 Roboflow 的 COCO JSON，并返回 ``(json路径, 图片目录)``。

    ``input_path`` 可以是：

    - Roboflow 导出根目录；
    - 包含 ``_annotations.coco.json`` 的 ``train`` 目录；
    - ``_annotations.coco.json`` 文件本身。
    """

    expanded_input = input_path.expanduser()
    if expanded_input.is_symlink():
        raise ValueError(
            f"Roboflow 输入不能是符号链接：{expanded_input.absolute()}"
        )
    input_path = expanded_input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在：{input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError(f"输入文件不是 JSON：{input_path}")
        return input_path, input_path.parent

    symlinks = []
    for path in input_path.rglob("*"):
        if path.is_symlink():
            symlinks.append(path)
    symlinks.sort()
    if symlinks:
        raise ValueError(
            "Roboflow 输入目录不能包含符号链接；请复制真实文件后再转换："
            f"{symlinks}"
        )

    exact_candidates = [
        input_path / "_annotations.coco.json",
        input_path / "train" / "_annotations.coco.json",
    ]
    existing_exact = []
    for path in exact_candidates:
        if path.is_file():
            existing_exact.append(path.resolve())
    if len(existing_exact) == 1:
        annotation_path = existing_exact[0]
        return annotation_path, annotation_path.parent

    candidates = []
    for path in input_path.rglob("_annotations.coco.json"):
        candidates.append(path.resolve())
    candidates.sort()
    if not candidates:
        for path in input_path.rglob("*.json"):
            candidates.append(path.resolve())
        candidates.sort()

    if not candidates:
        raise FileNotFoundError(f"在 {input_path} 下没有找到 COCO JSON。")
    if len(candidates) > 1:
        candidate_lines = []
        for path in candidates:
            candidate_lines.append(f"  - {path}")
        candidate_text = "\n".join(candidate_lines)
        raise ValueError(
            "发现多个 COCO JSON，无法确定应使用哪一个。请把 --input 指向具体 JSON "
            f"或对应 split 目录：\n{candidate_text}"
        )

    annotation_path = candidates[0]
    return annotation_path, annotation_path.parent


def _load_coco_json(annotation_path: Path) -> dict[str, Any]:
    """读取 JSON，并检查 COCO 的三个核心列表是否存在。"""

    data = json.loads(
        annotation_path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
    )
    if not isinstance(data, dict):
        raise ValueError(f"COCO 根节点必须是对象：{annotation_path}")

    for key in ("images", "annotations", "categories"):
        if not isinstance(data.get(key), list):
            raise ValueError(f"COCO 字段 {key!r} 必须是列表：{annotation_path}")
    return data


def _integer_id(value: Any, field_name: str) -> int:
    """把 COCO ID 转成整数，并拒绝布尔值和非整数形式。"""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} 不能是布尔值。")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 不是有效整数：{value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field_name} 不是整数：{value!r}")
    return integer


def _build_category_mapping(
    data: Mapping[str, Any],
    class_names: Sequence[str],
) -> tuple[dict[int, int], tuple[str, ...]]:
    """按类别名称建立 ``COCO category_id -> YOLO class_id`` 映射。

    不使用 ``category_id - 1`` 这类硬编码规则，因为 Roboflow 可能加入
    ``New-HPS`` 这样的空父类别，也可能使用不连续的 category ID。
    """

    normalized_name_list = []
    for name in class_names:
        clean_name = name.strip()
        if clean_name:
            normalized_name_list.append(clean_name)
    normalized_names = tuple(normalized_name_list)
    if not normalized_names:
        raise ValueError("类别列表不能为空。")
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError(f"类别列表中存在重复名称：{normalized_names}")

    class_map = load_class_map(normalized_names)
    category_name_by_id: dict[int, str] = {}
    category_id_by_name: dict[str, int] = {}

    for category in data["categories"]:
        if not isinstance(category, Mapping):
            raise ValueError(f"COCO category 必须是对象：{category!r}")
        category_id = _integer_id(category.get("id"), "category.id")
        name = str(category.get("name", "")).strip()
        if not name:
            raise ValueError(f"category_id={category_id} 缺少类别名称。")
        if category_id in category_name_by_id:
            raise ValueError(f"重复的 category_id：{category_id}")
        if name in category_id_by_name:
            raise ValueError(f"重复的 category name：{name}")
        category_name_by_id[category_id] = name
        category_id_by_name[name] = category_id

    missing_names = []
    for name in normalized_names:
        if name not in category_id_by_name:
            missing_names.append(name)
    if missing_names:
        raise ValueError(f"COCO categories 中缺少 classes.txt 类别：{missing_names}")

    category_counts: Counter[int] = Counter()
    for annotation in data["annotations"]:
        if not isinstance(annotation, Mapping):
            raise ValueError(f"COCO annotation 必须是对象：{annotation!r}")
        category_id = _integer_id(annotation.get("category_id"), "annotation.category_id")
        if category_id not in category_name_by_id:
            raise ValueError(f"标注引用了不存在的 category_id：{category_id}")
        category_counts[category_id] += 1

    mapping = {}
    for name, yolo_id in class_map.items():
        category_id = category_id_by_name[name]
        mapping[category_id] = yolo_id

    unknown_annotated = []
    for category_id, count in category_counts.items():
        if count > 0 and category_id not in mapping:
            unknown_annotated.append(category_name_by_id[category_id])
    if unknown_annotated:
        raise ValueError(
            "以下类别含有标注，但不在 classes.txt 中。为避免静默丢标，转换已停止："
            f"{sorted(unknown_annotated)}"
        )

    ignored_category_list = []
    for category_id, name in category_name_by_id.items():
        if category_id not in mapping and category_counts[category_id] == 0:
            ignored_category_list.append(name)
    ignored_category_list.sort()
    ignored_categories = tuple(ignored_category_list)
    return mapping, ignored_categories


def _source_group(image: Mapping[str, Any]) -> str:
    """读取 Roboflow 保存的原始图片名，作为防泄漏分组键。"""

    extra = image.get("extra")
    if not isinstance(extra, Mapping):
        raise ValueError(
            f"image_id={image.get('id')} 缺少 extra.name，无法按原图分组。"
        )

    group_name = str(extra.get("name", "")).strip()
    if not group_name:
        raise ValueError(
            f"image_id={image.get('id')} 的 extra.name 为空，无法按原图分组。"
        )
    return Path(group_name).name


def _safe_source_path(image_dir: Path, file_name: str) -> Path:
    """安全解析 COCO file_name，防止绝对路径或 ``..`` 跳出图片目录。"""

    relative_path = Path(file_name)
    if relative_path.is_absolute():
        raise ValueError(f"COCO file_name 不允许使用绝对路径：{file_name}")

    image_dir = image_dir.resolve()
    source_path = (image_dir / relative_path).resolve()
    try:
        source_path.relative_to(image_dir)
    except ValueError as exc:
        raise ValueError(f"COCO file_name 跳出了图片目录：{file_name}") from exc

    if not source_path.is_file():
        raise FileNotFoundError(f"COCO 引用的图片不存在：{source_path}")
    if source_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"不支持的图片格式：{source_path}")
    return source_path


def _validate_decoded_image(
    source_path: Path,
    expected_width: int,
    expected_height: int,
) -> None:
    """解码图片，并核对真实尺寸与 COCO 元数据。"""

    image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"图片无法解码：{source_path}")

    actual_height, actual_width = image.shape[:2]
    if (actual_width, actual_height) != (expected_width, expected_height):
        raise ValueError(
            f"图片尺寸与 COCO 不一致：{source_path.name}，"
            f"实际={actual_width}x{actual_height}，"
            f"COCO={expected_width}x{expected_height}"
        )


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    """使用鞋带公式计算多边形面积。"""

    area_twice = 0.0
    for index in range(len(points)):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % len(points)]
        area_twice += x1 * y2 - x2 * y1
    return abs(area_twice) * 0.5


def _normalize_polygon(
    segmentation: Any,
    image_width: int,
    image_height: int,
    annotation_id: int,
) -> tuple[str, ...]:
    """校验一个 COCO polygon，并生成 YOLO 所需的归一化坐标。"""

    if isinstance(segmentation, Mapping):
        raise ValueError(
            f"annotation_id={annotation_id} 使用 RLE，当前转换器只支持 polygon。"
        )
    if not isinstance(segmentation, list) or not segmentation:
        raise ValueError(f"annotation_id={annotation_id} 没有有效 segmentation。")
    if len(segmentation) != 1:
        raise ValueError(
            f"annotation_id={annotation_id} 含 {len(segmentation)} 个分离 polygon；"
            "YOLO 单实例无法无损表达，转换已停止。"
        )

    raw_polygon = segmentation[0]
    if not isinstance(raw_polygon, list):
        raise ValueError(f"annotation_id={annotation_id} 的 polygon 不是坐标列表。")
    if len(raw_polygon) < 6 or len(raw_polygon) % 2 != 0:
        raise ValueError(
            f"annotation_id={annotation_id} 的 polygon 坐标数必须是偶数且不少于 6。"
        )

    points: list[tuple[float, float]] = []
    for index in range(0, len(raw_polygon), 2):
        try:
            x = float(raw_polygon[index])
            y = float(raw_polygon[index + 1])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"annotation_id={annotation_id} 含非数值 polygon 坐标。"
            ) from exc

        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(
                f"annotation_id={annotation_id} 含 NaN 或无穷大坐标。"
            )
        if x < 0.0 or x > image_width or y < 0.0 or y > image_height:
            raise ValueError(
                f"annotation_id={annotation_id} 的坐标 ({x}, {y}) 超出"
                f"图像范围 0..{image_width}, 0..{image_height}。"
            )

        # 去掉相邻重复点，避免生成冗余或退化的 YOLO polygon。
        if not points or not (
            math.isclose(points[-1][0], x, abs_tol=1e-9)
            and math.isclose(points[-1][1], y, abs_tol=1e-9)
        ):
            points.append((x, y))

    # Roboflow 导出的 polygon 通常显式重复首点；YOLO 会自动闭合。
    if len(points) > 1 and (
        math.isclose(points[0][0], points[-1][0], abs_tol=1e-9)
        and math.isclose(points[0][1], points[-1][1], abs_tol=1e-9)
    ):
        points.pop()

    unique_points: set[tuple[float, float]] = set()
    for x, y in points:
        rounded_point = (round(x, 9), round(y, 9))
        unique_points.add(rounded_point)
    if len(points) < 3 or len(unique_points) < 3:
        raise ValueError(f"annotation_id={annotation_id} 少于 3 个有效多边形点。")
    if _polygon_area(points) < 1.0:
        raise ValueError(
            f"annotation_id={annotation_id} 的 polygon 面积小于 1 像素，"
            "无法形成稳定实例掩膜。"
        )

    normalized: list[str] = []
    for x, y in points:
        normalized.extend(
            [
                f"{min(max(x / image_width, 0.0), 1.0):.8f}",
                f"{min(max(y / image_height, 0.0), 1.0):.8f}",
            ]
        )
    return tuple(normalized)


def _prepare_images(
    data: Mapping[str, Any],
    image_dir: Path,
    category_mapping: Mapping[int, int],
) -> list[_PreparedImage]:
    """完成所有图片和标注的预检查，并在写文件前生成 YOLO 标签。"""

    image_by_id: dict[int, Mapping[str, Any]] = {}
    output_names: set[str] = set()
    output_stems: set[str] = set()

    for image in data["images"]:
        if not isinstance(image, Mapping):
            raise ValueError(f"COCO image 必须是对象：{image!r}")
        image_id = _integer_id(image.get("id"), "image.id")
        if image_id in image_by_id:
            raise ValueError(f"重复的 image_id：{image_id}")

        file_name = str(image.get("file_name", "")).strip()
        if not file_name:
            raise ValueError(f"image_id={image_id} 缺少 file_name。")
        output_name = Path(file_name).name
        output_stem = Path(output_name).stem
        if output_name in output_names:
            raise ValueError(f"输出图片文件名重复：{output_name}")
        if output_stem in output_stems:
            raise ValueError(f"输出标签文件名主体重复：{output_stem}")

        image_by_id[image_id] = image
        output_names.add(output_name)
        output_stems.add(output_stem)

    annotations_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    annotation_ids: set[int] = set()
    for annotation in data["annotations"]:
        if not isinstance(annotation, Mapping):
            raise ValueError(f"COCO annotation 必须是对象：{annotation!r}")
        annotation_id = _integer_id(annotation.get("id"), "annotation.id")
        if annotation_id in annotation_ids:
            raise ValueError(f"重复的 annotation_id：{annotation_id}")
        annotation_ids.add(annotation_id)

        image_id = _integer_id(annotation.get("image_id"), "annotation.image_id")
        if image_id not in image_by_id:
            raise ValueError(
                f"annotation_id={annotation_id} 引用了不存在的 image_id={image_id}。"
            )
        annotations_by_image[image_id].append(annotation)

    prepared: list[_PreparedImage] = []
    for image_id in sorted(image_by_id):
        image = image_by_id[image_id]
        width = _integer_id(image.get("width"), f"image_id={image_id}.width")
        height = _integer_id(image.get("height"), f"image_id={image_id}.height")
        if width <= 0 or height <= 0:
            raise ValueError(f"image_id={image_id} 的宽高必须大于 0。")

        file_name = str(image["file_name"]).strip()
        source_path = _safe_source_path(image_dir, file_name)
        _validate_decoded_image(source_path, width, height)

        label_lines: list[str] = []
        annotations = list(annotations_by_image.get(image_id, []))

        # sort 的 key 参数需要一个函数。这里不用 lambda，单独写出普通函数，
        # 初学时更容易看出：排序依据是每条标注的 annotation.id。
        def annotation_id_for_sort(annotation: Mapping[str, Any]) -> int:
            return _integer_id(annotation.get("id"), "annotation.id")

        annotations.sort(key=annotation_id_for_sort)
        for annotation in annotations:
            annotation_id = _integer_id(annotation.get("id"), "annotation.id")
            category_id = _integer_id(
                annotation.get("category_id"),
                f"annotation_id={annotation_id}.category_id",
            )
            if category_id not in category_mapping:
                # 带标注的未知类别已在类别映射阶段报错；这里用于防御性检查。
                raise ValueError(
                    f"annotation_id={annotation_id} 没有对应的 YOLO 类别。"
                )
            if _integer_id(annotation.get("iscrowd", 0), "annotation.iscrowd") != 0:
                raise ValueError(
                    f"annotation_id={annotation_id} 是 iscrowd 标注，当前不支持。"
                )

            coordinates = _normalize_polygon(
                annotation.get("segmentation"),
                width,
                height,
                annotation_id,
            )
            label_lines.append(
                f"{category_mapping[category_id]} " + " ".join(coordinates)
            )

        prepared.append(
            _PreparedImage(
                image_id=image_id,
                source_path=source_path,
                output_name=Path(file_name).name,
                source_group=_source_group(image),
                label_lines=tuple(label_lines),
            )
        )

    if not prepared:
        raise ValueError("COCO 数据集中没有图片。")
    return prepared


def _split_source_groups(
    prepared: Sequence[_PreparedImage],
    val_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """按原图组划分 train/val，保证同源增强图不会跨 split。"""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio 必须大于 0 且小于 1。")

    group_set: set[str] = set()
    for item in prepared:
        group_set.add(item.source_group)
    groups = sorted(group_set)
    if len(groups) < 2:
        raise ValueError("至少需要 2 个原图组才能建立训练集和验证集。")

    rng = random.Random(seed)
    rng.shuffle(groups)
    val_group_count = min(len(groups) - 1, max(1, round(len(groups) * val_ratio)))
    val_groups = set(groups[:val_group_count])
    train_groups = set(groups[val_group_count:])

    if train_groups & val_groups:
        raise AssertionError("内部错误：训练组和验证组发生重叠。")
    return train_groups, val_groups


def _check_overwrite_target(output_dir: Path) -> None:
    """拒绝覆盖过于宽泛或危险的目录。"""

    resolved = output_dir.resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if resolved in forbidden or len(resolved.parts) < 4:
        raise ValueError(f"拒绝覆盖危险输出目录：{resolved}")


def _reject_nested_input_output(
    input_path: Path,
    output_dir: Path,
) -> None:
    """拒绝输入与输出互相包含，避免生成文件污染或覆盖源导出。"""

    resolved_input = input_path.expanduser().resolve()
    if resolved_input.is_file():
        source_scope = resolved_input.parent
    else:
        source_scope = resolved_input
    resolved_output = output_dir.expanduser().resolve()
    if (
        resolved_output == source_scope
        or resolved_output in source_scope.parents
        or source_scope in resolved_output.parents
    ):
        raise ValueError(
            "Roboflow 输入与 YOLO 输出目录不能相同或互相嵌套："
            f"input={source_scope}，output={resolved_output}"
        )


def _replace_output_directory(
    temporary_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    """发布已验证数据集；未授权覆盖时绝不搬走并发生成的结果。"""

    backup_dir: Path | None = None
    if output_dir.exists() and not overwrite:
        try:
            output_dir.rmdir()
        except OSError as exc:
            raise FileExistsError(
                f"输出目录在转换期间变为非空，拒绝替换：{output_dir}"
            ) from exc
    elif output_dir.exists():
        backup_dir = output_dir.parent / (
            f".{output_dir.name}.backup-{uuid.uuid4().hex}"
        )
        output_dir.rename(backup_dir)
    try:
        temporary_dir.rename(output_dir)
    except Exception:
        if (
            backup_dir is not None
            and backup_dir.exists()
            and not output_dir.exists()
        ):
            backup_dir.rename(output_dir)
        raise
    else:
        if backup_dir is not None:
            shutil.rmtree(backup_dir)


def _validate_generated_dataset(
    dataset_dir: Path,
    class_count: int,
) -> tuple[int, int, Counter[int]]:
    """后验检查生成的图片、标签、类别编号和归一化坐标。"""

    image_total = 0
    label_total = 0
    class_counts: Counter[int] = Counter()

    for split in ("train", "val"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        image_stems: set[str] = set()
        for path in image_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                image_stems.add(path.stem)

        label_stems: set[str] = set()
        for path in label_dir.iterdir():
            if path.is_file() and path.suffix == ".txt":
                label_stems.add(path.stem)
        if image_stems != label_stems:
            missing_labels = sorted(image_stems - label_stems)
            orphan_labels = sorted(label_stems - image_stems)
            raise ValueError(
                f"{split} 图片与标签不匹配："
                f"缺少标签={missing_labels}，孤立标签={orphan_labels}"
            )

        image_total += len(image_stems)
        for label_path in sorted(label_dir.glob("*.txt")):
            seen_instance_boxes: set[tuple[float | int, ...]] = set()
            raw_lines = label_path.read_text(encoding="utf-8").splitlines()
            for line_index in range(len(raw_lines)):
                line_number = line_index + 1
                raw_line = raw_lines[line_index]
                line = raw_line.strip()
                if not line:
                    continue
                tokens = line.split()
                if len(tokens) < 7 or len(tokens) % 2 == 0:
                    raise ValueError(
                        f"YOLO 标签格式错误：{label_path}:{line_number}"
                    )
                try:
                    class_id = int(tokens[0])
                    coordinates: list[float] = []
                    for value in tokens[1:]:
                        coordinates.append(float(value))
                except ValueError as exc:
                    raise ValueError(
                        f"YOLO 标签含非数值字段：{label_path}:{line_number}"
                    ) from exc
                if not 0 <= class_id < class_count:
                    raise ValueError(
                        f"YOLO 类别越界：{label_path}:{line_number} class={class_id}"
                    )
                coordinates_are_valid = True
                for value in coordinates:
                    if not math.isfinite(value) or value < 0.0 or value > 1.0:
                        coordinates_are_valid = False
                        break
                if not coordinates_are_valid:
                    raise ValueError(
                        f"YOLO 坐标越界：{label_path}:{line_number}"
                    )
                # 输入坐标会格式化为 8 位小数；即使原始 polygon 面积大于 0，
                # 也必须确认序列化后没有因舍入而塌成重复点或零面积图形。
                points: list[tuple[float, float]] = []
                for coordinate_index in range(0, len(coordinates), 2):
                    x = coordinates[coordinate_index]
                    y = coordinates[coordinate_index + 1]
                    points.append((x, y))

                unique_points: set[tuple[float, float]] = set()
                x_values: list[float] = []
                y_values: list[float] = []
                for x, y in points:
                    unique_points.add((round(x, 9), round(y, 9)))
                    x_values.append(x)
                    y_values.append(y)
                if len(unique_points) < 3 or _polygon_area(points) <= 0.0:
                    raise ValueError(
                        f"YOLO polygon 退化：{label_path}:{line_number}"
                    )
                instance_box = (
                    class_id,
                    round(min(x_values), 9),
                    round(min(y_values), 9),
                    round(max(x_values), 9),
                    round(max(y_values), 9),
                )
                if instance_box in seen_instance_boxes:
                    raise ValueError(
                        "YOLO 标签包含会被 Ultralytics 静默去重的同类别同框实例："
                        f"{label_path}:{line_number}"
                    )
                seen_instance_boxes.add(instance_box)
                label_total += 1
                class_counts[class_id] += 1

    return image_total, label_total, class_counts


def convert_roboflow_coco_dataset(
    input_path: Path,
    output_dir: Path,
    class_names: Sequence[str],
    val_ratio: float = 0.2,
    seed: int = 42,
    overwrite: bool = False,
) -> RoboflowConversionStats:
    """执行一次完整的 Roboflow COCO → YOLO Seg 转换。"""

    annotation_path, image_dir = find_coco_annotation(input_path)
    image_dir_symlinks: list[Path] = []
    for path in image_dir.rglob("*"):
        if path.is_symlink():
            image_dir_symlinks.append(path)
    image_dir_symlinks.sort()
    if image_dir_symlinks:
        raise ValueError(
            "Roboflow 图片/标注目录不能包含符号链接："
            f"{image_dir_symlinks}"
        )
    _reject_nested_input_output(input_path, output_dir)
    data = _load_coco_json(annotation_path)
    category_mapping, ignored_categories = _build_category_mapping(data, class_names)
    prepared = _prepare_images(data, image_dir, category_mapping)
    train_groups, val_groups = _split_source_groups(prepared, val_ratio, seed)

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"输出路径已存在且不是目录：{output_dir}")
    output_has_files = False
    if output_dir.exists():
        for unused_path in output_dir.iterdir():
            output_has_files = True
            break
    if output_has_files and not overwrite:
        raise FileExistsError(
            f"输出目录非空：{output_dir}。请改用新目录，或明确添加 --overwrite。"
        )
    if overwrite:
        _check_overwrite_target(output_dir)

    # 先在同级临时目录完整生成并验证，成功后再一次性替换目标目录。
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )

    split_image_counts: Counter[str] = Counter()
    split_label_counts: Counter[str] = Counter()
    split_class_counts: dict[str, Counter[int]] = {
        "train": Counter(),
        "val": Counter(),
    }

    try:
        for split in ("train", "val"):
            (temporary_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (temporary_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        for item in prepared:
            if item.source_group in val_groups:
                split = "val"
            else:
                split = "train"
            target_image = temporary_dir / "images" / split / item.output_name
            target_label = temporary_dir / "labels" / split / f"{Path(item.output_name).stem}.txt"

            shutil.copy2(item.source_path, target_image)
            label_text = "\n".join(item.label_lines)
            if item.label_lines:
                label_text += "\n"
            target_label.write_text(label_text, encoding="utf-8")

            split_image_counts[split] += 1
            split_label_counts[split] += len(item.label_lines)
            for line in item.label_lines:
                split_class_counts[split][int(line.split(maxsplit=1)[0])] += 1

        write_dataset_yaml(
            output_dir,
            class_names,
            yaml_path=temporary_dir / "tube_seg.yaml",
        )

        stats = RoboflowConversionStats(
            images=len(prepared),
            annotations=len(data["annotations"]),
            labels_written=sum(split_label_counts.values()),
            source_groups=len(train_groups | val_groups),
            train_groups=len(train_groups),
            val_groups=len(val_groups),
            train_images=split_image_counts["train"],
            val_images=split_image_counts["val"],
            train_labels=split_label_counts["train"],
            val_labels=split_label_counts["val"],
        )

        train_class_counts: dict[str, int] = {}
        val_class_counts: dict[str, int] = {}
        for class_id in range(len(class_names)):
            class_id_text = str(class_id)
            train_class_counts[class_id_text] = split_class_counts["train"][class_id]
            val_class_counts[class_id_text] = split_class_counts["val"][class_id]

        report = {
            "source_annotation": str(annotation_path),
            "output_dir": str(output_dir),
            "classes": list(class_names),
            "ignored_empty_categories": list(ignored_categories),
            "seed": seed,
            "val_ratio": val_ratio,
            "stats": asdict(stats),
            "splits": {
                "train": {
                    "source_groups": sorted(train_groups),
                    "class_counts": train_class_counts,
                },
                "val": {
                    "source_groups": sorted(val_groups),
                    "class_counts": val_class_counts,
                },
            },
        }
        (temporary_dir / "conversion_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        image_total, label_total, class_counts = _validate_generated_dataset(
            temporary_dir,
            len(class_names),
        )
        if image_total != stats.images:
            raise AssertionError(
                f"生成图片数不一致：预期={stats.images}，实际={image_total}"
            )
        if label_total != stats.labels_written:
            raise AssertionError(
                f"生成标签数不一致：预期={stats.labels_written}，实际={label_total}"
            )
        if sum(class_counts.values()) != stats.labels_written:
            raise AssertionError("生成后的类别计数与标签总数不一致。")

        # 再次检查并发写入：未授权覆盖时，转换期间出现的新文件也不能被替换。
        output_became_nonempty = False
        if output_dir.exists():
            for unused_path in output_dir.iterdir():
                output_became_nonempty = True
                break
        if output_became_nonempty and not overwrite:
            raise FileExistsError(f"输出目录在转换期间变为非空：{output_dir}")
        _replace_output_directory(
            temporary_dir,
            output_dir,
            overwrite=overwrite,
        )
        return stats
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise


# =============================================================================
# 命令行入口
# =============================================================================

def parse_args() -> argparse.Namespace:
    """读取 Roboflow 转换参数。"""

    parser = argparse.ArgumentParser(
        description="把 Roboflow COCO 分割数据转换为 YOLO Seg 数据集。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Roboflow 导出根目录、train 目录或 COCO JSON。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets" / "tube_seg_roboflow",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=ROOT / "configs" / "classes.txt",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    """执行 Roboflow COCO 到 YOLO Seg 的完整转换。"""

    args = parse_args()
    class_names = class_names_from_file(args.classes)
    stats = convert_roboflow_coco_dataset(
        input_path=args.input,
        output_dir=args.output,
        class_names=class_names,
        val_ratio=args.val_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    output_dir = args.output.expanduser().resolve()
    print("Roboflow COCO -> YOLO Seg 转换完成")
    print(stats)
    print(f"Dataset YAML: {output_dir / 'tube_seg.yaml'}")
    print(f"Conversion report: {output_dir / 'conversion_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
