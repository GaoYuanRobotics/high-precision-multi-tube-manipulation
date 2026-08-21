#!/usr/bin/env python3
"""第 01 步：检查当前 Python 环境是否满足项目依赖。

默认检查 NumPy、OpenCV、RealSense 和 CArm 等基础依赖；添加
``--with-yolo`` 后，再检查 PyTorch、Ultralytics 及 CUDA 状态。

本脚本只读取环境并打印报告，不安装软件、不打开相机、也不连接机械臂。
初学者建议先读文件末尾的 ``main()``，再看 ``check_packages()``。
"""

from __future__ import annotations

import argparse
import importlib
import sys
from importlib import metadata


# =============================================================================
# 1. 项目要求的依赖和固定版本
# =============================================================================

# 每项依次是：Python 导入名、pip 包名、必须精确匹配的版本。
# None 表示只检查能否导入，不限制具体版本。
CORE_MODULES = [
    ("numpy", "numpy", None),
    ("scipy", "scipy", None),
    ("cv2", "opencv-python", None),
    ("yaml", "pyyaml", None),
    ("tqdm", "tqdm", None),
    ("pyrealsense2", "pyrealsense2", None),
    # CArm 接口直接关系真实动作，因此固定到已经验证的 SDK 版本。
    ("carm", "carm", "0.1.20260512"),
]

YOLO_MODULES = [
    ("torch", "torch", "2.12.0+cu130"),
    ("torchvision", "torchvision", "0.27.0+cu130"),
    ("ultralytics", "ultralytics", "8.4.56"),
]


# =============================================================================
# 2. 环境检查
# =============================================================================

def installed_version(module, package_name: str) -> str:
    """读取发行包版本；没有标准元数据时退回模块的 __version__。"""

    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown"))


def check_packages(packages) -> list[str]:
    """逐个导入依赖，返回需要重新安装的 pip requirement 列表。"""

    missing: list[str] = []
    for module_name, package_name, required_version in packages:
        if required_version is None:
            requirement = package_name
        else:
            requirement = f"{package_name}=={required_version}"
        try:
            # 二进制库即使已经安装，也可能因为动态链接库问题而无法导入。
            module = importlib.import_module(module_name)
            current_version = installed_version(module, package_name)
        except Exception as exc:
            print(f"[missing] {module_name}: {exc}")
            missing.append(requirement)
            continue

        if required_version is not None and current_version != required_version:
            print(
                f"[version mismatch] {module_name}: "
                f"installed={current_version}, required={required_version}"
            )
            missing.append(requirement)
        else:
            print(f"[ok] {module_name}: {current_version}")
    return missing


def print_cuda_status() -> None:
    """打印 PyTorch 实际看到的 CUDA 构建版本和 GPU 数量。"""

    # 只有用户要求 YOLO 检查时才加载体积较大的 PyTorch。
    import torch

    print(f"\nTorch CUDA available: {torch.cuda.is_available()}")
    print(f"Torch CUDA version: {torch.version.cuda}")
    print(f"Torch device count: {torch.cuda.device_count()}")


# =============================================================================
# 3. 命令行和主流程
# =============================================================================

def parse_args() -> argparse.Namespace:
    """读取终端参数，并返回 argparse.Namespace。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-yolo",
        action="store_true",
        help="同时检查 torch、torchvision、ultralytics 和 CUDA 状态。",
    )
    return parser.parse_args()


def main() -> int:
    """检查用户请求的依赖；全部通过返回 0，否则返回 1。"""

    args = parse_args()
    packages = list(CORE_MODULES)
    if args.with_yolo:
        packages.extend(YOLO_MODULES)

    print(f"Python: {sys.version.split()[0]}")
    missing = check_packages(packages)
    if missing:
        print("\nInstall missing or mismatched packages:")
        print("pip install " + " ".join(sorted(set(missing))))
        return 1

    if args.with_yolo:
        print_cuda_status()
    else:
        print(
            "\nYOLO check skipped. "
            "Run with --with-yolo after installing requirements-yolo.txt."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
