#!/usr/bin/env python3
"""第 04 步：把 X-AnyLabeling polygon 标注转换成 YOLO Seg 数据集。

本文件同时包含数据校验、格式转换、数据集划分和命令行入口。默认拒绝覆盖
非空输出目录，只有明确添加 ``--overwrite`` 才会
替换已有数据。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import yaml


# 项目根目录只用于提供默认输入输出路径，不会修改 Python 搜索路径。
ROOT = Path(__file__).resolve().parents[1]


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class ConversionStats:
    """完成 ConversionStats 对应的单一处理步骤。"""

    images: int
    train: int
    val: int
    labels: int
    # 为兼容早期调用保留；转换器现在对任何坏 shape 都会报错并停止，所以恒为 0。
    skipped_shapes: int


def _unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """拒绝 JSON 重复键，避免后一个关键字段静默覆盖前一个。"""

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key is not allowed: {key!r}.")
        result[key] = value
    return result


def _positive_json_integer(value: object, field_name: str) -> int:
    """只接受真正的正整数，拒绝 bool、浮点截断和数字字符串。"""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive JSON integer.")
    return value


def load_class_map(names: Sequence[str]) -> dict[str, int]:
    """完成 load_class_map 对应的单一处理步骤。"""

    if not names:
        raise ValueError("At least one class name is required.")
    normalized = []
    for name in names:
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("Class names cannot be empty.")
        normalized.append(clean_name)

    seen_names = set()
    duplicate_names = set()
    for name in normalized:
        if name in seen_names:
            duplicate_names.add(name)
        seen_names.add(name)
    if duplicate_names:
        duplicates = sorted(duplicate_names)
        raise ValueError(f"Duplicate class names are not allowed: {duplicates}")

    class_map = {}
    class_id = 0
    for name in normalized:
        class_map[name] = class_id
        class_id += 1
    return class_map


def iter_image_json_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    """完成 iter_image_json_pairs 对应的单一处理步骤。"""

    symlinks = []
    images = []
    for path in input_dir.rglob("*"):
        if path.is_symlink():
            symlinks.append(path)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
    symlinks.sort()
    images.sort()
    if symlinks:
        raise ValueError(
            "Input annotations must not contain symbolic links; copy the "
            f"source files into the input tree instead: {symlinks}"
        )
    missing_json = []
    for image_path in images:
        if not image_path.with_suffix(".json").is_file():
            missing_json.append(image_path)
    if missing_json:
        raise ValueError(
            "Images without same-stem X-AnyLabeling JSON are not allowed: "
            f"{missing_json}"
        )

    pairs: list[tuple[Path, Path]] = []
    for image_path in images:
        # X-AnyLabeling/LabelMe 默认是 image.jpg + image.json 这种一一对应。
        json_path = image_path.with_suffix(".json")
        pairs.append((image_path, json_path))

    expected_json = set()
    for _, json_path in pairs:
        expected_json.add(json_path.resolve())

    orphan_json = []
    for path in input_dir.rglob("*.json"):
        if path.resolve() not in expected_json:
            orphan_json.append(path)
    orphan_json.sort()
    if orphan_json:
        raise ValueError(
            "Annotation JSON files without a same-stem supported image are not "
            f"allowed: {orphan_json}"
        )
    return pairs


def convert_shape_to_yolo_line(
    shape: Mapping,
    class_map: Mapping[str, int],
    image_width: int,
    image_height: int,
) -> str:
    """完成 convert_shape_to_yolo_line 对应的单一处理步骤。"""

    shape_type = str(shape.get("shape_type", "")).strip().lower()
    if shape_type != "polygon":
        raise ValueError(
            "shape_type must be 'polygon' for YOLO segmentation, got "
            f"{shape_type or '<missing>'!r}."
        )
    label = str(shape.get("label", ""))
    if label not in class_map:
        raise ValueError(
            f"Unknown class label {label!r}; expected one of "
            f"{sorted(class_map)}."
        )

    points = shape.get("points") or []
    if (
        isinstance(points, (str, bytes))
        or not isinstance(points, Sequence)
        or len(points) < 3
    ):
        raise ValueError("Polygon must contain at least three coordinate pairs.")

    normalized: list[str] = []
    pixel_points: list[tuple[float, float]] = []
    for point in points:
        if (
            isinstance(point, (str, bytes))
            or not isinstance(point, Sequence)
            or len(point) != 2
        ):
            raise ValueError(
                "Every polygon point must contain exactly two coordinates."
            )
        try:
            pixel_x = float(point[0])
            pixel_y = float(point[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("Polygon contains a non-numeric coordinate.") from exc
        if not math.isfinite(pixel_x) or not math.isfinite(pixel_y):
            raise ValueError("Polygon contains NaN or infinity.")
        if (
            pixel_x < 0.0
            or pixel_x > image_width
            or pixel_y < 0.0
            or pixel_y > image_height
        ):
            raise ValueError(
                "Polygon coordinate is outside the image bounds: "
                f"({pixel_x}, {pixel_y}) not in "
                f"0..{image_width}, 0..{image_height}."
            )
        pixel_points.append((pixel_x, pixel_y))
        # YOLO segmentation 要求坐标归一化到 0-1。
        # 坐标已在上面严格验证，不再 clamp；静默裁剪可能把图外 polygon 压成
        # 全部位于 0/1 边界的退化标签。
        x = pixel_x / image_width
        y = pixel_y / image_height
        normalized.extend([f"{x:.8f}", f"{y:.8f}"])

    # 三个点也可能完全共线；这种零面积 polygon 不能形成有效实例 mask。
    twice_area = 0.0
    point_count = len(pixel_points)
    for index in range(point_count):
        x1, y1 = pixel_points[index]
        next_index = (index + 1) % point_count
        x2, y2 = pixel_points[next_index]
        twice_area += x1 * y2 - x2 * y1
    twice_area = abs(twice_area)
    if twice_area < 2.0:
        raise ValueError("Polygon area is zero or too small.")
    return f"{class_map[label]} " + " ".join(normalized)


def convert_one_json(
    json_path: Path,
    class_map: Mapping[str, int],
    expected_image_size: tuple[int, int] | None = None,
    expected_image_name: str | None = None,
) -> tuple[list[str], int]:
    """完成 convert_one_json 对应的单一处理步骤。"""

    data = json.loads(
        json_path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
    )
    if not isinstance(data, Mapping):
        raise ValueError(f"{json_path} top level must be an object.")
    if expected_image_name is not None and "imagePath" in data:
        raw_image_path = data["imagePath"]
        if (
            not isinstance(raw_image_path, str)
            or not raw_image_path.strip()
            or Path(raw_image_path).name != expected_image_name
        ):
            raise ValueError(
                f"{json_path} imagePath={raw_image_path!r} does not match "
                f"the paired image {expected_image_name!r}."
            )
    try:
        image_width = _positive_json_integer(
            data.get("imageWidth"),
            "imageWidth",
        )
        image_height = _positive_json_integer(
            data.get("imageHeight"),
            "imageHeight",
        )
    except ValueError as exc:
        raise ValueError(f"{json_path} has invalid dimensions: {exc}") from exc
    if (
        expected_image_size is not None
        and (image_width, image_height) != expected_image_size
    ):
        raise ValueError(
            f"{json_path} metadata size {image_width}x{image_height} does not "
            f"match the real image {expected_image_size[0]}x{expected_image_size[1]}."
        )

    lines: list[str] = []
    shapes = data.get("shapes", [])
    if not isinstance(shapes, list):
        raise ValueError(f"{json_path} shapes must be a list.")
    shape_index = 0
    for shape in shapes:
        if not isinstance(shape, Mapping):
            raise ValueError(
                f"{json_path} shape[{shape_index}] must be an object."
            )
        try:
            line = convert_shape_to_yolo_line(
                shape,
                class_map,
                image_width,
                image_height,
            )
        except ValueError as exc:
            raise ValueError(
                f"{json_path} shape[{shape_index}] is invalid: {exc}"
            ) from exc
        lines.append(line)
        shape_index += 1
    # 第二个返回值只为兼容旧接口；坏 shape 现在会 fail-fast，不再跳过。
    return lines, 0


def write_dataset_yaml(
    dataset_dir: Path,
    class_names: Sequence[str],
    yaml_path: Path | None = None,
) -> Path:
    """完成 write_dataset_yaml 对应的单一处理步骤。"""

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


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    """使用鞋带公式计算多边形的绝对面积。"""

    doubled_area = 0.0
    point_count = len(points)
    for index in range(point_count):
        x1, y1 = points[index]
        next_index = (index + 1) % point_count
        x2, y2 = points[next_index]
        doubled_area += x1 * y2 - x2 * y1
    return 0.5 * abs(doubled_area)


def _validate_generated_dataset(
    dataset_dir: Path,
    class_count: int,
) -> tuple[int, int, Counter[int]]:
    """再次检查图片/标签配对和每一个生成的 YOLO 多边形。"""

    image_total = 0
    label_total = 0
    class_counts: Counter[int] = Counter()

    for split in ("train", "val"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        image_stems = set()
        for path in image_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                image_stems.add(path.stem)

        label_stems = set()
        for path in label_dir.iterdir():
            if path.is_file() and path.suffix.lower() == ".txt":
                label_stems.add(path.stem)
        if image_stems != label_stems:
            raise ValueError(
                f"{split} image/label mismatch: "
                f"missing labels={sorted(image_stems - label_stems)}, "
                f"orphan labels={sorted(label_stems - image_stems)}."
            )

        image_total += len(image_stems)
        for label_path in sorted(label_dir.glob("*.txt")):
            seen_instance_boxes: set[tuple[float | int, ...]] = set()
            line_number = 0
            raw_lines = label_path.read_text(encoding="utf-8").splitlines()
            for raw_line in raw_lines:
                line_number += 1
                line = raw_line.strip()
                if not line:
                    continue
                tokens = line.split()
                if len(tokens) < 7 or len(tokens) % 2 == 0:
                    raise ValueError(
                        f"Invalid YOLO polygon format: "
                        f"{label_path}:{line_number}."
                    )
                try:
                    class_id = int(tokens[0])
                    coordinates = []
                    for value in tokens[1:]:
                        coordinates.append(float(value))
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric YOLO label field: "
                        f"{label_path}:{line_number}."
                    ) from exc
                if not 0 <= class_id < class_count:
                    raise ValueError(
                        f"YOLO class id out of range: "
                        f"{label_path}:{line_number} class={class_id}."
                    )
                coordinates_are_valid = True
                for value in coordinates:
                    if not math.isfinite(value) or value < 0.0 or value > 1.0:
                        coordinates_are_valid = False
                        break
                if not coordinates_are_valid:
                    raise ValueError(
                        f"YOLO coordinate out of range: "
                        f"{label_path}:{line_number}."
                    )

                points = []
                for coordinate_index in range(0, len(coordinates), 2):
                    x = coordinates[coordinate_index]
                    y = coordinates[coordinate_index + 1]
                    points.append((x, y))

                unique_points = set()
                x_values = []
                y_values = []
                for x, y in points:
                    unique_points.add((round(x, 9), round(y, 9)))
                    x_values.append(x)
                    y_values.append(y)
                if len(unique_points) < 3 or _polygon_area(points) <= 0.0:
                    raise ValueError(
                        f"Degenerate YOLO polygon: "
                        f"{label_path}:{line_number}."
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
                        "Duplicate class/bounding-box instance would be "
                        f"silently removed by Ultralytics: "
                        f"{label_path}:{line_number}."
                    )
                seen_instance_boxes.add(instance_box)
                label_total += 1
                class_counts[class_id] += 1

    return image_total, label_total, class_counts


def _replace_output_directory(
    temporary_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    """发布已验证的数据集，并避免静默覆盖并发生成的结果。"""

    backup_dir: Path | None = None
    if output_dir.exists() and not overwrite:
        try:
            output_dir.rmdir()
        except OSError as exc:
            raise FileExistsError(
                "Output directory became non-empty during conversion; "
                f"refusing to replace it: {output_dir}"
            ) from exc
    elif output_dir.exists():
        backup_dir = output_dir.parent / (
            f".{output_dir.name}.backup-{uuid.uuid4().hex}"
        )
        output_dir.rename(backup_dir)

    try:
        temporary_dir.rename(output_dir)
    except Exception:
        if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            backup_dir.rename(output_dir)
        raise
    else:
        if backup_dir is not None:
            shutil.rmtree(backup_dir)


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    class_names: Sequence[str],
    val_ratio: float = 0.2,
    seed: int = 42,
    overwrite: bool = False,
) -> ConversionStats:
    """完成 convert_dataset 对应的单一处理步骤。"""

    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not 0.0 < float(val_ratio) < 1.0:
        raise ValueError("val_ratio must be greater than 0 and smaller than 1.")
    if (
        output_dir == input_dir
        or output_dir in input_dir.parents
        or input_dir in output_dir.parents
    ):
        raise ValueError(
            "Input and output directories must not be identical or nested."
        )

    pairs = iter_image_json_pairs(input_dir)
    if not pairs:
        raise FileNotFoundError(f"No image/json pairs found in {input_dir}.")
    if len(pairs) < 2:
        raise ValueError(
            "At least two image/JSON pairs are required to create non-empty "
            "train and val splits."
        )

    # 扁平输出使用原文件名，因此预先拒绝递归目录中的同名图片/标签，避免覆盖。
    image_names = []
    label_names = []
    for image_path, _ in pairs:
        image_names.append(image_path.name.casefold())
        label_names.append(image_path.stem.casefold())
    if len(set(image_names)) != len(image_names):
        raise ValueError(
            "Recursive input contains duplicate image filenames; flattening "
            "would overwrite files. Rename them before conversion."
        )
    if len(set(label_names)) != len(label_names):
        raise ValueError(
            "Recursive input contains duplicate image stems; YOLO label "
            "filenames would collide."
        )

    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(
            f"Output path exists and is not a directory: {output_dir}"
        )
    output_is_nonempty = False
    if output_dir.exists():
        for _ in output_dir.iterdir():
            output_is_nonempty = True
            break
    if output_is_nonempty and not overwrite:
        raise ValueError(
            f"Output directory is not empty: {output_dir}. "
            "Use --overwrite only after checking the path."
        )
    if output_is_nonempty and overwrite:
        forbidden = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
        if output_dir in forbidden or len(output_dir.parts) < 4:
            raise ValueError(f"Refusing to overwrite dangerous path: {output_dir}")

    rng = random.Random(seed)
    rng.shuffle(pairs)

    # 固定 seed，保证你每次转换出来的 train/val 划分一致。
    val_count = min(len(pairs) - 1, max(1, round(len(pairs) * val_ratio)))
    val_set = set(pairs[:val_count])
    class_map = load_class_map(class_names)

    # 先在输出目录同级的临时目录完整生成并后验校验。只有全部成功后才交换目录，
    # 因此坏 JSON、坏图片或坏标签不会破坏现有的可用数据集。
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )

    labels_written = 0
    skipped_shapes = 0
    train_count = 0
    val_count_actual = 0

    try:
        for split in ("train", "val"):
            (temporary_dir / "images" / split).mkdir(
                parents=True,
                exist_ok=True,
            )
            (temporary_dir / "labels" / split).mkdir(
                parents=True,
                exist_ok=True,
            )

        for image_path, json_path in pairs:
            if (image_path, json_path) in val_set:
                split = "val"
            else:
                split = "train"
            if split == "train":
                train_count += 1
            else:
                val_count_actual += 1

            target_image = (
                temporary_dir / "images" / split / image_path.name
            )
            target_label = (
                temporary_dir
                / "labels"
                / split
                / f"{image_path.stem}.txt"
            )

            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Cannot decode image: {image_path}")
            actual_height, actual_width = image.shape[:2]
            lines, skipped = convert_one_json(
                json_path,
                class_map,
                expected_image_size=(actual_width, actual_height),
                expected_image_name=image_path.name,
            )
            shutil.copy2(image_path, target_image)
            label_text = "\n".join(lines)
            if lines:
                label_text += "\n"
            target_label.write_text(label_text, encoding="utf-8")
            labels_written += len(lines)
            skipped_shapes += skipped

        write_dataset_yaml(
            output_dir,
            class_names,
            yaml_path=temporary_dir / "tube_seg.yaml",
        )
        image_total, label_total, _ = _validate_generated_dataset(
            temporary_dir,
            len(class_names),
        )
        if image_total != len(pairs):
            raise AssertionError(
                f"Generated image count mismatch: "
                f"expected={len(pairs)}, actual={image_total}."
            )
        if label_total != labels_written:
            raise AssertionError(
                f"Generated label count mismatch: "
                f"expected={labels_written}, actual={label_total}."
            )

        _replace_output_directory(
            temporary_dir,
            output_dir,
            overwrite=overwrite,
        )
        return ConversionStats(
            images=len(pairs),
            train=train_count,
            val=val_count_actual,
            labels=labels_written,
            skipped_shapes=skipped_shapes,
        )
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise


def class_names_from_file(path: Path) -> list[str]:
    """完成 class_names_from_file 对应的单一处理步骤。"""

    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        clean_line = line.strip()
        if clean_line and not clean_line.startswith("#"):
            names.append(clean_line)
    if not names:
        raise ValueError(f"No class names found in {path}.")
    return names


def class_names_from_text(values: Iterable[str]) -> list[str]:
    """完成 class_names_from_text 对应的单一处理步骤。"""

    names = []
    for value in values:
        clean_value = value.strip()
        if clean_value:
            names.append(clean_value)
    if not names:
        raise ValueError("No class names provided.")
    return names


# =============================================================================
# 命令行入口
# =============================================================================

def parse_args() -> argparse.Namespace:
    """读取用户在终端输入的转换参数。"""

    parser = argparse.ArgumentParser(description="X-AnyLabeling 转 YOLO Seg 数据集")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="包含图片及其同名 X-AnyLabeling JSON 的目录。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets" / "tube_seg",
        help="输出目录，默认 datasets/tube_seg。",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=ROOT / "configs" / "classes.txt",
        help="类别文件；有效行顺序决定 YOLO 类别 ID。",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="明确替换已有非空输出目录；默认拒绝覆盖。",
    )
    return parser.parse_args()


def main() -> int:
    """执行 X-AnyLabeling 到 YOLO Seg 的完整转换。"""

    args = parse_args()
    class_names = class_names_from_file(args.classes)
    stats = convert_dataset(
        args.input,
        args.output,
        class_names,
        args.val_ratio,
        args.seed,
        overwrite=args.overwrite,
    )
    print(stats)
    print(f"Dataset YAML: {(args.output / 'tube_seg.yaml').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
