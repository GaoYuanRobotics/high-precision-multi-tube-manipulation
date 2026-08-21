#!/usr/bin/env python3
"""第 06 步：训练四类试管部件的 YOLO26 实例分割模型。

脚本读取第 04 步（X-AnyLabeling）或第 05 步（Roboflow COCO）生成的 YOLO
分割数据集，并以 YOLO26 实例分割预训练权重为起点进行迁移学习。当前默认使用
Roboflow 数据集 ``datasets/tube_seg_roboflow/tube_seg.yaml`` 和
``yolo26x-seg.pt``，训练以下四类：

- ``p-body``：紫色试管管身；
- ``p-cap``：紫色试管盖；
- ``y-body``：黄色试管管身；
- ``y-cap``：黄色试管盖。

当前四类中没有 ``rack_top``，因此本模型不负责分割试管架。

典型运行命令：

    python scripts/06_train_yolo26_seg.py \
      --data datasets/tube_seg_roboflow/tube_seg.yaml \
      --model yolo26x-seg.pt \
      --epochs 100 \
      --imgsz 1024 \
      --aug-profile roboflow-light \
      --device 0

Ultralytics 会在本次实验目录的 ``weights`` 子目录同时保存 ``best.pt`` 和
``last.pt``。用于后续预览与几何计算的应是验证指标最好的 ``best.pt``，而不是
最后一轮的 ``last.pt``。默认的新训练结果保存到：

    runs/segment/tube_seg/weights/best.pt

实际位置仍以训练开始时终端打印的 ``save_dir`` 为准。修复默认输出路径之前产生
的历史模型仍保留在下列旧嵌套目录，不需要移动或删除：

    runs/segment/runs/segment/tube_seg/weights/best.pt

主流程：读取参数 -> 完整预检数据集 -> 加载分割模型 -> 选择增强档位
-> 启动训练。数据或参数不合格时会在加载模型和占用 GPU 之前停止。

本脚本不采集图片、不转换标注。初学者建议先读 ``main()``，再看
``train_model()``；较长的 ``validate_training_inputs()`` 是训练前数据门禁。
"""

# 启用较新的类型注解规则，让复杂类型注解在运行时延迟求值。
from __future__ import annotations

# argparse 用于读取终端中的训练选项。
import argparse
# hashlib 用 SHA-256 比较 train/val 图片的真实解码像素，检查数据泄漏。
import hashlib
# math 提供 isfinite()，用于拒绝 NaN 和正负无穷坐标。
import math
# Counter 可以方便地统计是否存在大小写冲突的重复文件主体。
from collections import Counter

# Path 用于表示数据集 YAML 等本地路径。
from pathlib import Path

# PyYAML 用于读取 YOLO 数据集配置；下方使用自定义安全加载器拒绝重复键。
import yaml


# 项目根目录用于构造绝对输出路径，避免 Ultralytics 8.4.56 把相对的
# ``runs/segment`` 再拼接到全局 runs_dir，产生 ``runs/segment/runs/segment``。
ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# 1. 三档在线增强和固定数据契约
# =============================================================================

# 三档在线增强配置：
#
# roboflow-light：
#   供已经在 Roboflow 做过离线增强的数据使用，也是当前默认值。它关闭容易再次
#   产生黑边或拼接画面的旋转、翻转和 Mosaic，只保留少量平移、缩放及光照变化。
#
# none：
#   关闭所有显式在线增强，适合排查训练问题或做严格对照实验。
#
# strong：
#   保留项目原先的强增强设置，适合仅使用未增强原图时尝试；不建议与 Roboflow
#   增强后的数据叠加。
AUGMENTATION_PROFILES: dict[str, dict[str, float | int]] = {
    # 轻量档：保护颜色类别，只做小幅位置、尺度、饱和度和亮度扰动。
    "roboflow-light": {
        # 几何增强：角度、平移比例、缩放幅度和水平/垂直翻转概率。
        "degrees": 0.0,
        "translate": 0.02,
        "scale": 0.10,
        "fliplr": 0.0,
        "flipud": 0.0,
        # 四类标签依赖紫色/黄色，因此不改变色相，只轻微改变饱和度和亮度。
        "hsv_h": 0.0,
        "hsv_s": 0.15,
        "hsv_v": 0.10,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        # 0 表示无需在最后若干轮额外关闭 Mosaic，因为本档从未开启它。
        "close_mosaic": 0,
    },
    # 对照档：所有显式在线增强均关闭。
    "none": {
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 0,
    },
    # 强增强档：可以任意旋转并使用 Mosaic，仅建议未增强原始数据。
    "strong": {
        "degrees": 180.0,
        "translate": 0.05,
        "scale": 0.30,
        "fliplr": 0.5,
        "flipud": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 10,
    },
}

# 模型和数据集必须严格遵守这个类别 ID 顺序，不能只保证名字“都存在”。
EXPECTED_CLASSES = ["p-body", "p-cap", "y-body", "y-cap"]
# 与 requirements-yolo.txt 固定的 Ultralytics 8.4.56 ``IMG_FORMATS`` 对齐。
# 训练前预检（preflight）必须扫描训练器可能加载的全部顶层图片格式；否则坏的
# HEIC/DNG 之类文件会绕过这里，直到模型/GPU 已加载后才失败。
IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".dng",
    ".heic",
    ".heif",
    ".jp2",
    ".jpeg",
    ".jpeg2000",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """拒绝数据集 YAML 重复键，避免路径或类别被后一个值静默覆盖。

    普通 YAML 加载器遇到两个同名键时通常保留后一个，例如两个 ``train``。
    训练配置属于安全边界，因此本项目把这种含糊配置直接视为错误。
    """


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    """逐项构造 YAML 字典，并在写入前检查键是否已经出现。"""

    # mapping 保存当前 YAML 对象已经解析出的键和值。
    mapping: dict[object, object] = {}
    # node.value 中每一项都是尚未转换成 Python 对象的“键节点、值节点”。
    for key_node, value_node in node.value:
        # 先解析键，才能判断它是否重复。
        key = loader.construct_object(key_node, deep=deep)
        try:
            # 字典查找同时验证该键可以被哈希。
            duplicate = key in mapping
        except TypeError as exc:
            raise ValueError("数据集 YAML 的映射键必须可哈希。") from exc
        # 重复键会使配置含义不唯一，因此不允许继续加载。
        if duplicate:
            raise ValueError(f"数据集 YAML 含重复键：{key!r}。")
        # 只有通过检查后才解析并保存对应的值。
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


# 告诉 PyYAML：以后遇到普通 mapping（字典）节点时，使用上面的严格构造器。
_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


# =============================================================================
# 2. 命令行
# =============================================================================

def parse_args() -> argparse.Namespace:
    """定义并解析第 06 步模型训练所需的命令行参数。"""

    # description 会显示在 ``python ... --help`` 的开头。
    parser = argparse.ArgumentParser(
        description=(
            "训练 p-body、p-cap、y-body、y-cap 四类 YOLO26 实例分割模型。"
        )
    )

    # 数据集配置文件由第 04/05 步的 X-AnyLabeling 或 Roboflow 转换脚本生成。
    # 该 YAML 文件记录训练集、验证集路径以及类别名称。
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "datasets" / "tube_seg_roboflow" / "tube_seg.yaml",
        help=(
            "YOLO 数据集 YAML 路径，默认是 "
            "datasets/tube_seg_roboflow/tube_seg.yaml。"
        ),
    )

    # 初始模型权重。当前默认使用 YOLO26 extra-large 分割预训练模型。
    # 也可以传入之前训练得到的 best.pt，从已有权重继续训练。
    parser.add_argument(
        "--model",
        default="yolo26x-seg.pt",
        help=(
            "初始分割模型名称或本地权重路径，默认是 yolo26x-seg.pt；"
            "本地不存在时 Ultralytics 会自动下载官方预训练权重。"
        ),
    )

    # epoch 表示模型完整遍历一次训练集。
    # 这里设置的是最大训练轮数；如果触发早停，实际轮数可能少于该值。
    # 当前数据只有 9 个独立原图组，默认先使用 100 轮完成第一版实验。
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="最大训练轮数，默认是 100。",
    )

    # 训练前，Ultralytics 会把图片缩放到指定尺寸。
    # 1024 能保留较多边界细节，但比 640 占用更多显存和训练时间。
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="训练图像尺寸，默认是 1024。",
    )

    # batch 是每个训练批次同时处理的图片数量。
    # 如果出现 CUDA out of memory，可以尝试把它从 8 降到 4、2 或 1。
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="每批图片数量，默认是 8；显存不足时请调小。",
    )

    # "0" 表示使用第 1 块 CUDA 显卡；"1" 表示第 2 块显卡；
    # 如果没有可用显卡，也可以显式传入 "cpu"，但训练速度会明显降低。
    parser.add_argument(
        "--device",
        default="0",
        help="训练设备，默认 0 表示第 1 块 CUDA 显卡，也可以使用 cpu。",
    )

    # DataLoader 使用的后台进程数量。适当增加可以提高图片读取速度，
    # 但会占用更多 CPU 和内存。
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="加载训练数据的后台进程数，默认是 4。",
    )

    # project 是所有训练实验的上级输出目录。
    parser.add_argument(
        "--project",
        type=Path,
        default=ROOT / "runs" / "segment",
        help=(
            "训练实验的上级目录，默认使用项目内绝对路径 runs/segment；"
            "最终绝对保存位置以终端打印的 save_dir 为准。"
        ),
    )

    # name 是本次实验的名称，通常会成为 project 下的子目录名。
    parser.add_argument(
        "--name",
        default="tube_seg",
        help="本次实验名称，默认 tube_seg；通常成为 save_dir 的末级目录。",
    )

    # 在线增强档位。当前默认使用适合 Roboflow 离线增强数据的轻量配置。
    parser.add_argument(
        "--aug-profile",
        choices=tuple(AUGMENTATION_PROFILES),
        default="roboflow-light",
        help=(
            "在线增强档位：roboflow-light（默认，适合已增强数据）、"
            "none（关闭增强）或 strong（仅建议未增强原图使用）。"
        ),
    )

    # 读取用户传入的参数并返回 Namespace 对象。
    return parser.parse_args()


# =============================================================================
# 3. 训练前数据预检
# =============================================================================

def validate_training_inputs(args: argparse.Namespace) -> list[str]:
    """在加载大模型或占用 GPU 之前执行训练前预检（preflight）。

    预检会检查训练参数、YAML 类别契约、train/val 目录、图片解码、标签多边形
    和跨集合像素泄漏。成功时返回按类别 ID 排列的名称列表；遇到任何不确定或
    危险的数据结构都会抛出异常。把检查放在 ``YOLO(...)`` 之前，可以避免
    花费显存和时间后才发现数据有问题。
    """

    # epoch、输入尺寸和 batch 至少都要是 1。
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0:
        raise ValueError("--epochs、--imgsz 和 --batch 必须是正整数。")
    # workers=0 合法，表示在主进程加载；负数没有意义。
    if args.workers < 0:
        raise ValueError("--workers 不能小于 0。")
    # 实验名最终会成为一个目录名，所以拒绝空字符串、"."、".." 和路径。
    if (
        not isinstance(args.name, str)
        or not args.name.strip()
        or Path(args.name).name != args.name
        or args.name in {".", ".."}
    ):
        raise ValueError("--name 必须是单个非空实验目录名，不能包含路径分隔符。")

    # 把用户目录符号 ``~`` 展开，并固定为绝对路径，便于后续安全比较。
    dataset_yaml = args.data.expanduser().resolve()
    # is_file() 同时排除不存在路径和误传进来的目录。
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"找不到数据集 YAML：{dataset_yaml}")
    try:
        # 使用拒绝重复键的 SafeLoader；read_text 明确按 UTF-8 读取中文类别。
        data = yaml.load(
            dataset_yaml.read_text(encoding="utf-8"),
            Loader=_UniqueKeySafeLoader,
        )
    # 将 YAML 语法错误和本项目严格加载器的错误统一转成清晰提示。
    except (ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"无法解析数据集 YAML：{dataset_yaml}：{exc}") from exc
    # 数据集配置最外层应当是键值对象，而不是列表、字符串或空值。
    if not isinstance(data, dict):
        raise ValueError(f"数据集 YAML 顶层必须是对象：{dataset_yaml}")

    # Ultralytics 接受两种 names 写法：
    #   names: [p-body, p-cap, ...]
    # 或 names: {0: p-body, 1: p-cap, ...}
    raw_names = data.get("names")
    if isinstance(raw_names, list):
        # 转成字符串列表，后面使用同一种数据类型做精确顺序比较。
        class_names: list[str] = []
        for name in raw_names:
            class_names.append(str(name))
    elif isinstance(raw_names, dict):
        # bool 是 int 的子类，所以必须额外排除 True/False 这类伪类别 ID。
        keys_are_valid = True
        for key in raw_names:
            if isinstance(key, bool) or not isinstance(key, int):
                keys_are_valid = False
                break
        if not keys_are_valid:
            raise ValueError("数据集 names 的键必须是真正的整数类别 ID。")
        # 保留“类别编号 -> 名称”关系，随后检查编号是否从 0 连续增长。
        indexed: dict[int, str] = {}
        for key, value in raw_names.items():
            indexed[key] = str(value)
        if sorted(indexed) != list(range(len(indexed))):
            raise ValueError("数据集 names 的类别 ID 必须从 0 开始连续排列。")
        # 按 ID 重新组成列表，避免依赖 YAML 中字典键的书写顺序。
        class_names = []
        for index in range(len(indexed)):
            class_names.append(indexed[index])
    else:
        raise ValueError("数据集 YAML 缺少 names 列表或映射。")
    # 这里使用列表精确比较：名称相同但顺序不同也会导致标签语义错位。
    if class_names != EXPECTED_CLASSES:
        raise ValueError(
            "数据集类别或顺序不符合当前四类模型："
            f"实际={class_names}，期望={EXPECTED_CLASSES}。"
        )
    # nc 是可选的类别数量；一旦写出，就必须是真正的整数 4。
    if "nc" in data:
        raw_nc = data["nc"]
        if (
            isinstance(raw_nc, bool)
            or not isinstance(raw_nc, int)
            or raw_nc != len(class_names)
        ):
            raise ValueError(
                f"数据集 nc={raw_nc!r} 必须是与 names 数量 "
                f"{len(class_names)} 一致的整数。"
            )

    # path 是 train/val 相对路径共同参照的数据集根目录；省略时取 YAML 所在目录。
    raw_root_value = data.get("path", str(dataset_yaml.parent))
    # 路径只能是非空字符串，拒绝列表、数字等会被 Path 含糊解释的类型。
    if not isinstance(raw_root_value, str) or not raw_root_value.strip():
        raise ValueError("数据集 path 必须是非空路径字符串。")
    raw_root = Path(raw_root_value).expanduser()
    # 绝对 path 直接解析；相对 path 则相对于 YAML 文件所在目录解析。
    if raw_root.is_absolute():
        dataset_root = raw_root.resolve()
    else:
        dataset_root = (dataset_yaml.parent / raw_root).resolve()
    # 保存 train/val 各类别实例数量，最后打印给用户核对。
    split_class_counts: dict[str, list[int]] = {}
    # 保存“解码像素 SHA-256 -> 文件路径”，用于检测 train/val 完全相同图片。
    split_image_hashes: dict[str, dict[str, Path]] = {
        "train": {},
        "val": {},
    }
    # 训练集和验证集执行完全相同的结构、图片和标签检查。
    for split in ("train", "val"):
        # YAML 中的 train/val 必须各自给出一个目录路径。
        raw_split = data.get(split)
        if not isinstance(raw_split, str) or not raw_split.strip():
            raise ValueError(f"数据集 YAML 缺少非空的 {split} 路径。")
        split_path = Path(raw_split).expanduser()
        # 相对 split 以 dataset_root 为基准，绝对 split 则直接解析。
        if split_path.is_absolute():
            split_path = split_path.resolve()
        else:
            split_path = (dataset_root / split_path).resolve()
        try:
            # relative_to() 成功表示 split 确实位于 dataset_root 内部。
            split_path.relative_to(dataset_root)
        except ValueError as exc:
            raise ValueError(
                f"数据集 {split} 路径必须位于 dataset path 内：{split_path}"
            ) from exc
        # 后续会遍历目录，因此这里明确要求它存在且是目录。
        if not split_path.is_dir():
            raise FileNotFoundError(
                f"数据集 {split} 路径不存在或不是目录：{split_path}"
            )
        # 本项目采用标准 YOLO 目录：images/train 与 images/val。
        if split_path.parent.name != "images" or split_path.name != split:
            raise ValueError(
                f"数据集 {split} 必须使用标准 images/{split} 目录：{split_path}"
            )
        # 从 images/<split> 的同级结构推导对应 labels/<split>。
        labels_path = (
            split_path.parent.parent / "labels" / split_path.name
        )
        if not labels_path.is_dir():
            raise FileNotFoundError(
                f"数据集 {split} 标签目录不存在：{labels_path}"
            )

        # Ultralytics 会递归扫描 images/<split>。本项目的两个转换器输出扁平
        # split；若这里仍只检查顶层文件，嵌套目录中的坏图或坏标签会绕过
        # 训练前预检，直到模型加载/GPU 占用后才失败。为保持预检集合与训练集合
        # 完全一致，当前契约明确拒绝子目录和符号链接。
        for directory, description in (
            (split_path, "图片"),
            (labels_path, "标签"),
        ):
            # iterdir() 只检查当前层；任何真实子目录都违反扁平数据集契约。
            nested_directories: list[Path] = []
            for entry in directory.iterdir():
                if entry.is_dir():
                    nested_directories.append(entry)
            nested_directories.sort()
            # 符号链接可能把训练器引向数据集根目录之外，因此全部拒绝。
            symbolic_links: list[Path] = []
            for entry in directory.iterdir():
                if entry.is_symlink():
                    symbolic_links.append(entry)
            symbolic_links.sort()
            if nested_directories:
                raise ValueError(
                    f"数据集 {split} {description}目录必须是扁平结构，"
                    f"不允许子目录：{nested_directories}"
                )
            if symbolic_links:
                raise ValueError(
                    f"数据集 {split} {description}目录不允许符号链接："
                    f"{symbolic_links}"
                )

        # 只收集与固定 Ultralytics 版本支持格式一致的顶层普通图片文件。
        images: list[Path] = []
        for path in split_path.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                images.append(path)
        images.sort()
        # 一个 YOLO 分割标签就是同名的 .txt；空标签文件本身允许存在。
        labels = sorted(labels_path.glob("*.txt"))
        if not images:
            raise ValueError(f"数据集 {split} 没有可训练图片。")
        # casefold() 比 lower() 更适合做不区分大小写的文件名冲突检查。
        image_stem_keys: list[str] = []
        for path in images:
            image_stem_keys.append(path.stem.casefold())
        label_stem_keys: list[str] = []
        for path in labels:
            label_stem_keys.append(path.stem.casefold())
        # Counter 记录每个主体名出现次数，超过 1 就代表同名或大小写冲突。
        image_stem_counts = Counter(image_stem_keys)
        label_stem_counts = Counter(label_stem_keys)
        duplicate_image_stems: list[str] = []
        for stem, count in image_stem_counts.items():
            if count > 1:
                duplicate_image_stems.append(stem)
        duplicate_image_stems.sort()

        duplicate_label_stems: list[str] = []
        for stem, count in label_stem_counts.items():
            if count > 1:
                duplicate_label_stems.append(stem)
        duplicate_label_stems.sort()
        if duplicate_image_stems:
            raise ValueError(
                f"数据集 {split} 图片文件主体重复（含大小写冲突）："
                f"{duplicate_image_stems}。每张图片必须独占一个标签文件。"
            )
        if duplicate_label_stems:
            raise ValueError(
                f"数据集 {split} 标签文件主体重复（含大小写冲突）："
                f"{duplicate_label_stems}。"
            )
        # 大小写折叠只用于冲突检测；Linux 上训练器按原始大小写寻找标签，
        # 因此 Foo.jpg 绝不能被 foo.txt 误判为已配对。
        image_stems: set[str] = set()
        for path in images:
            image_stems.add(path.stem)
        label_stems: set[str] = set()
        for path in labels:
            label_stems.add(path.stem)
        if image_stems != label_stems:
            raise ValueError(
                f"数据集 {split} 图片/标签不一一对应："
                f"缺标签={sorted(image_stems - label_stems)}，"
                f"孤立标签={sorted(label_stems - image_stems)}。"
            )

        # 训练前解码每张图片，避免模型初始化和占用 GPU 后才发现损坏文件。
        import cv2

        for image_path in images:
            # IMREAD_UNCHANGED 保留图片实际通道数和位深，既检查可解码性，也让
            # 后面的像素哈希包含真实数据类型信息。
            decoded = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if decoded is None:
                raise ValueError(f"数据集图片无法解码：{image_path}")
            # 仅哈希原始字节会让不同形状/类型但字节相同的数组产生歧义，所以
            # 依次加入 shape、dtype 和按 C 顺序排列的全部解码像素。
            digest = hashlib.sha256()
            digest.update(str(decoded.shape).encode("ascii"))
            digest.update(decoded.dtype.str.encode("ascii"))
            digest.update(decoded.tobytes(order="C"))
            split_image_hashes[split][digest.hexdigest()] = image_path

        # counts 的索引就是 class_id，值是当前 split 中该类实例总数。
        counts = [0] * len(class_names)
        # 每个 .txt 文件逐行检查；YOLO 中一行代表一个实例多边形。
        for label_path in labels:
            # 同一标签文件内，同类别且包围框完全相同的实例会被训练器去重。
            seen_instance_boxes: set[tuple[float | int, ...]] = set()
            raw_lines = label_path.read_text(encoding="utf-8").splitlines()
            for line_index in range(len(raw_lines)):
                line_number = line_index + 1
                raw_line = raw_lines[line_index]
                # 去掉行首尾空白；空行不表示实例，直接忽略。
                line = raw_line.strip()
                if not line:
                    continue
                # 一行格式为 class_id x1 y1 x2 y2 x3 y3 ...。
                tokens = line.split()
                # 最少是 1 个类别 + 3 个点的 6 个坐标，共 7 项；总项数应为奇数。
                if len(tokens) < 7 or len(tokens) % 2 == 0:
                    raise ValueError(
                        f"YOLO 分割标签格式错误：{label_path}:{line_number}"
                    )
                try:
                    # 第一个字段必须是整数类别，其余字段都转换为浮点坐标。
                    class_id = int(tokens[0])
                    coordinates: list[float] = []
                    for value in tokens[1:]:
                        coordinates.append(float(value))
                except ValueError as exc:
                    raise ValueError(
                        f"YOLO 标签含非数值字段：{label_path}:{line_number}"
                    ) from exc
                # 合法类别 ID 范围是 0 到类别数量减 1。
                if not 0 <= class_id < len(class_names):
                    raise ValueError(
                        f"YOLO 类别越界：{label_path}:{line_number}"
                    )
                # YOLO 坐标已经除以图像宽高，必须是 0..1 内的有限浮点数。
                coordinates_are_valid = True
                for value in coordinates:
                    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                        coordinates_are_valid = False
                        break
                if not coordinates_are_valid:
                    raise ValueError(
                        f"YOLO 坐标不是 0..1 的有限值："
                        f"{label_path}:{line_number}"
                    )
                # 把扁平坐标 [x1,y1,x2,y2,...] 两两配成 (x,y) 点。
                points: list[tuple[float, float]] = []
                for coordinate_index in range(0, len(coordinates), 2):
                    x = coordinates[coordinate_index]
                    y = coordinates[coordinate_index + 1]
                    points.append((x, y))
                # 九位小数统一浮点比较精度，检查是否至少有三个不同顶点。
                unique_points: set[tuple[float, float]] = set()
                x_values: list[float] = []
                y_values: list[float] = []
                for x, y in points:
                    unique_points.add((round(x, 9), round(y, 9)))
                    x_values.append(x)
                    y_values.append(y)
                # 鞋带公式计算多边形面积的两倍；结果为 0 表示所有点共线等退化形状。
                signed_area_twice = 0.0
                for point_index in range(len(points)):
                    x1, y1 = points[point_index]
                    next_index = (point_index + 1) % len(points)
                    x2, y2 = points[next_index]
                    signed_area_twice += x1 * y2 - x2 * y1
                area_twice = abs(signed_area_twice)
                if len(unique_points) < 3 or area_twice <= 0.0:
                    raise ValueError(
                        f"YOLO polygon 退化：{label_path}:{line_number}"
                    )
                # 用 class_id 和多边形的最小外接轴对齐框构造实例身份。
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
                # 记录该实例，并给对应类别计数加一。
                seen_instance_boxes.add(instance_box)
                counts[class_id] += 1
        # 每个 split 都必须包含四类，否则验证指标或训练语义会不完整。
        missing_classes: list[str] = []
        for index in range(len(counts)):
            count = counts[index]
            if count == 0:
                missing_classes.append(class_names[index])
        if missing_classes:
            raise ValueError(
                f"数据集 {split} 缺少类别实例：{missing_classes}。"
            )
        # 保存当前 split 的统计结果，供循环结束后统一打印。
        split_class_counts[split] = counts

    # 两个键集合的交集表示 train 和 val 中存在解码像素完全相同的图片。
    # 这只能发现“像素完全一致”的泄漏；不同增强版本不一定具有相同哈希，
    # 因此 Roboflow 数据仍必须先由脚本 05 按 extra.name 原图组划分。
    leaked_hashes = (
        set(split_image_hashes["train"])
        & set(split_image_hashes["val"])
    )
    if leaked_hashes:
        # 同时列出两侧路径，让用户可以回到转换步骤定位泄漏来源。
        examples: list[tuple[Path, Path]] = []
        for digest in sorted(leaked_hashes):
            train_path = split_image_hashes["train"][digest]
            val_path = split_image_hashes["val"][digest]
            examples.append((train_path, val_path))
        raise ValueError(
            "训练集与验证集包含解码像素完全相同的图片，存在数据泄漏："
            f"{examples}"
        )

    # 单元测试可能构造不含 model 的最小 Namespace，所以这里兼容属性缺失；
    # 正常命令行运行时该属性始终存在。
    model_argument = getattr(args, "model", None)
    if model_argument is not None and (
        not isinstance(model_argument, str) or not model_argument.strip()
    ):
        raise ValueError("--model 必须是非空模型名称或权重路径。")

    # 同理，若提供了 project，就检查输出位置不会污染数据集本身。
    project_value = getattr(args, "project", None)
    if project_value is not None:
        # 输出上级路径可以尚未创建，但若已经存在就必须是目录。
        project = Path(project_value).expanduser().resolve()
        if project.exists() and not project.is_dir():
            raise ValueError(f"--project 已存在但不是目录：{project}")
        experiment_dir = (project / args.name).resolve()
        try:
            # 若 relative_to 成功，说明训练输出误放在 dataset_root 内。
            experiment_dir.relative_to(dataset_root)
        except ValueError:
            # ValueError 在此处表示“不在数据集内”，正是安全情况。
            pass
        else:
            raise ValueError(
                "训练输出目录不能位于数据集目录内部："
                f"{experiment_dir}"
            )
    split_summaries: list[str] = []
    for split, counts in split_class_counts.items():
        class_count_map: dict[str, int] = {}
        for index in range(len(class_names)):
            class_name = class_names[index]
            class_count_map[class_name] = counts[index]
        split_summaries.append(f"{split}={class_count_map}")
    print("Dataset class instances: " + ", ".join(split_summaries))
    # 返回经过验证的类别顺序，供 main() 再次打印确认。
    return class_names


# =============================================================================
# 4. 模型训练
# =============================================================================

def train_model(args: argparse.Namespace) -> None:
    """加载分割模型，并把已经验证的参数交给 Ultralytics。"""

    # 延迟导入保证 --help 和数据预检不会提前加载大型深度学习库。
    from ultralytics import YOLO

    model = YOLO(args.model)
    if model.task != "segment":
        raise ValueError(
            f"--model 必须是实例分割模型，当前任务为 {model.task!r}。"
        )

    augmentation = AUGMENTATION_PROFILES[args.aug_profile]
    print(f"Augmentation profile: {args.aug_profile}")
    print(f"Augmentation settings: {augmentation}")

    model.train(
        data=str(args.data.expanduser().resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        # 整个项目统一使用 FP32；关闭 Ultralytics 默认可能启用的 AMP 混合精度。
        amp=False,
        # 传入绝对字符串可避开 Ultralytics 对相对 project 的二次拼接。
        project=str(args.project.expanduser().resolve()),
        name=args.name,
        seed=42,
        deterministic=True,
        patience=40,
        degrees=augmentation["degrees"],
        translate=augmentation["translate"],
        scale=augmentation["scale"],
        fliplr=augmentation["fliplr"],
        flipud=augmentation["flipud"],
        hsv_h=augmentation["hsv_h"],
        hsv_s=augmentation["hsv_s"],
        hsv_v=augmentation["hsv_v"],
        mosaic=augmentation["mosaic"],
        mixup=augmentation["mixup"],
        copy_paste=augmentation["copy_paste"],
        close_mosaic=augmentation["close_mosaic"],
        overlap_mask=True,
        plots=True,
    )


# =============================================================================
# 5. 主流程
# =============================================================================

def main() -> int:
    """按照本脚本的编号流程依次执行各个步骤。"""

    args = parse_args()
    class_names = validate_training_inputs(args)
    print(f"Validated dataset classes: {class_names}")
    train_model(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
