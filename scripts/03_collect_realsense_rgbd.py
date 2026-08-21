#!/usr/bin/env python3
"""第 03 步：采集 RealSense 彩色图和对齐深度图。

流程：检查参数 -> 严格选择相机 -> 创建全新会话 -> 保存相机标定信息
-> 实时预览 -> 按 ``s`` 保存一帧，按 ``q`` 退出。

默认输出：

    data/raw/<会话>/color/frame_000000.jpg
    data/raw/<会话>/depth/frame_000000.png
    data/raw/<会话>/depth_npy/frame_000000.npy
    data/raw/<会话>/meta/frame_000000.json
    data/raw/<会话>/intrinsics.json
    data/raw/<会话>/session.json

``--no-depth`` 只保存彩色图；``--auto-interval`` 可以定时保存。每帧文件
先写入临时目录，全部成功后才逐个发布，并且绝不覆盖已有文件。

初学者建议先读 ``main()`` 和 ``capture_loop()``，再学习安全写入部分。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# =============================================================================
# 1. 命令行和会话目录
# =============================================================================

def parse_args() -> argparse.Namespace:
    """读取终端参数，并返回 argparse.Namespace。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw"),
        help="所有采集会话的根目录，默认 data/raw。",
    )
    parser.add_argument(
        "--session",
        help="本次会话子目录名；默认使用 YYYYMMDD_HHMMSS 时间戳。",
    )
    parser.add_argument(
        "--serial",
        help="RealSense 序列号；连接多台相机时必须指定。",
    )
    parser.add_argument("--width", type=int, default=1280, help="图像宽度。")
    parser.add_argument("--height", type=int, default=720, help="图像高度。")
    parser.add_argument("--fps", type=int, default=30, help="相机帧率。")
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="只采集彩色图，不保存深度。",
    )
    parser.add_argument(
        "--auto-interval",
        type=float,
        default=0.0,
        help="每隔 N 秒保存一帧；0 表示关闭自动保存。",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Path:
    """检查尺寸、保存间隔和会话名称，返回不会覆盖旧数据的会话路径。"""

    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise ValueError("--width、--height 和 --fps 必须大于 0。")
    if not math.isfinite(args.auto_interval) or args.auto_interval < 0:
        raise ValueError("--auto-interval 必须是大于等于 0 的有限数值。")

    session_name = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")
    if (
        not isinstance(session_name, str)
        or not session_name.strip()
        or Path(session_name).name != session_name
        or session_name in {".", ".."}
    ):
        raise ValueError(
            "--session 必须是单个非空目录名，不能包含路径分隔符、'.' 或 '..'。"
        )

    output_root = args.out.expanduser().resolve()
    session_dir = output_root / session_name
    try:
        session_dir.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("--session 解析后超出 --out 目录，已拒绝。") from exc

    # lexists() 对失效符号链接也返回 True，避免把残留链接当成新目录。
    if os.path.lexists(session_dir):
        raise FileExistsError(
            f"采集会话目录已经存在，拒绝复用或覆盖：{session_dir}。"
            "请更换 --session 名称。"
        )
    return session_dir


# =============================================================================
# 2. RealSense 标定信息转换
# =============================================================================

def intrinsics_to_dict(intrinsics: Any) -> dict:
    """把 RealSense 内参对象转换成可以写入 JSON 的普通字典。"""

    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "model": str(intrinsics.model),
        "coeffs": list(intrinsics.coeffs),
    }


def extrinsics_to_dict(extrinsics: Any) -> dict:
    """保存深度光学系到彩色光学系的旋转和平移。

    librealsense 把九个旋转元素按列主序保存，所以使用 ``order='F'`` 还原
    3×3 数学矩阵。平移单位由 SDK 规定为米。
    """

    rotation_flat = np.asarray(extrinsics.rotation, dtype=np.float64)
    if rotation_flat.shape != (9,) or not np.isfinite(rotation_flat).all():
        raise ValueError("RealSense 外参 rotation 必须包含 9 个有限数值。")

    translation_m = np.asarray(extrinsics.translation, dtype=np.float64)
    if translation_m.shape != (3,) or not np.isfinite(translation_m).all():
        raise ValueError("RealSense 外参 translation 必须包含 3 个有限数值。")

    rotation_matrix = rotation_flat.reshape((3, 3), order="F")
    return {
        "rotation_matrix_row_major_3x3": rotation_matrix.tolist(),
        "librealsense_rotation_column_major_flat": rotation_flat.tolist(),
        "translation_m": translation_m.tolist(),
    }


def safe_device_info(device: Any, key: Any) -> str | None:
    """读取可选设备信息；设备或旧固件不支持时返回 None。"""

    try:
        if device.supports(key):
            return str(device.get_info(key))
    except RuntimeError:
        pass
    return None


# =============================================================================
# 3. 严格选择相机
# =============================================================================

def choose_camera_serial(
    requested_serial: str | None,
    available_serials: list[str | None],
) -> str:
    """选择唯一 RealSense；检测到多台时禁止自动猜测。"""

    requested = None
    if isinstance(requested_serial, str):
        stripped_serial = requested_serial.strip()
        if stripped_serial:
            requested = stripped_serial
    if requested_serial is not None and requested is None:
        raise ValueError("--serial 不能是空字符串。")

    serials = []
    for serial in available_serials:
        clean_serial = None
        if isinstance(serial, str):
            stripped_serial = serial.strip()
            if stripped_serial:
                clean_serial = stripped_serial
        serials.append(clean_serial)
    if requested is not None:
        if requested not in serials:
            visible = []
            for serial in serials:
                if serial is not None:
                    visible.append(serial)
            raise ValueError(
                f"--serial={requested!r} 不在当前 RealSense 设备中：{visible}"
            )
        return requested

    if not serials:
        raise RuntimeError("没有检测到 RealSense 设备。")
    if len(serials) > 1:
        visible = []
        for serial in serials:
            if serial:
                visible.append(serial)
            else:
                visible.append("<序列号不可用>")
        raise ValueError(
            "检测到多台 RealSense，必须显式提供 --serial，避免采错相机："
            f"{visible}"
        )
    if serials[0] is None:
        raise RuntimeError("唯一 RealSense 无法读取序列号，不能建立可追溯采集会话。")
    return serials[0]


# =============================================================================
# 4. 不覆盖旧文件的安全写入
# =============================================================================

def reserve_session_directory(session_dir: Path) -> None:
    """独占创建全新会话目录；已有目录即使为空也拒绝复用。"""

    session_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        session_dir.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"采集会话目录已经存在，拒绝复用或覆盖：{session_dir}。"
            "请更换 --session 名称。"
        ) from exc

    try:
        for child in ("color", "depth", "depth_npy", "meta"):
            (session_dir / child).mkdir(exist_ok=False)
    except Exception:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise


def commit_file_no_clobber(staged_path: Path, final_path: Path) -> None:
    """用同文件系统硬链接原子发布单个文件，并拒绝覆盖目标。"""

    try:
        os.link(staged_path, final_path)
    except FileExistsError as exc:
        raise FileExistsError(f"目标文件已存在，拒绝覆盖：{final_path}") from exc
    staged_path.unlink()


def write_json_no_clobber(path: Path, payload: dict) -> None:
    """完整写入并同步临时 JSON，再发布最终文件名。"""

    staged_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with staged_path.open("x", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
            file.flush()
            os.fsync(file.fileno())
        commit_file_no_clobber(staged_path, path)
    finally:
        staged_path.unlink(missing_ok=True)


def save_frame(
    session_dir: Path,
    index: int,
    color: np.ndarray,
    depth: np.ndarray | None,
    metadata: dict,
) -> None:
    """保存一组同编号 RGB-D 文件；任一步失败就回滚本次已发布文件。

    单个文件名通过硬链接原子发布，但一组 JPEG/PNG/NPY/JSON 不是一个文件系统
    事务。进程被强制终止或断电后，仍应检查该会话的文件配对完整性。
    """

    stem = f"frame_{index:06d}"
    final_paths = {
        "color": session_dir / "color" / f"{stem}.jpg",
        "meta": session_dir / "meta" / f"{stem}.json",
    }
    if depth is not None:
        final_paths.update(
            {
                "depth": session_dir / "depth" / f"{stem}.png",
                "depth_npy": session_dir / "depth_npy" / f"{stem}.npy",
            }
        )

    existing = []
    for path in final_paths.values():
        if path.exists():
            existing.append(path)
    if existing:
        raise FileExistsError(f"帧 {stem} 的目标文件已经存在，拒绝覆盖：{existing}")

    staging_root = session_dir / ".staging"
    staging_dir = staging_root / f"{stem}-{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=False)
    staged_paths = {
        "color": staging_dir / f"{stem}.jpg",
        "meta": staging_dir / f"{stem}.json",
    }
    if depth is not None:
        staged_paths.update(
            {
                "depth": staging_dir / f"{stem}.png",
                "depth_npy": staging_dir / f"{stem}.npy",
            }
        )

    # 记录设备号和 inode，回滚时只删除本次真正发布、且没有被替换的文件。
    committed: list[tuple[Path, int, int]] = []
    try:
        if not cv2.imwrite(str(staged_paths["color"]), color):
            raise OSError(f"彩色图写入失败：{final_paths['color']}")
        if depth is not None:
            if not cv2.imwrite(str(staged_paths["depth"]), depth):
                raise OSError(f"深度 PNG 写入失败：{final_paths['depth']}")
            np.save(staged_paths["depth_npy"], depth)
        staged_paths["meta"].write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        for key, final_path in final_paths.items():
            commit_file_no_clobber(staged_paths[key], final_path)
            stat = final_path.stat(follow_symlinks=False)
            committed.append((final_path, stat.st_dev, stat.st_ino))
    except Exception:
        for path, expected_device, expected_inode in committed:
            try:
                stat = path.stat(follow_symlinks=False)
                if stat.st_dev == expected_device and stat.st_ino == expected_inode:
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        try:
            staging_root.rmdir()
        except OSError:
            pass

    print(f"saved {stem}")


# =============================================================================
# 5. RealSense 会话
# =============================================================================

class RealSenseCapture:
    """管理一台 RealSense 的选择、启动、标定信息和资源释放。"""

    def __init__(self, args: argparse.Namespace, session_dir: Path):
        # 延迟导入：--help 和单元测试不需要加载真实相机 SDK。
        """初始化当前对象，并保存后续操作需要的状态。"""

        import pyrealsense2 as rs

        self.rs = rs
        self.args = args
        self.session_dir = session_dir
        self.pipeline = rs.pipeline()
        self.started = False
        self.align = None
        self.device_serial = ""

        devices = list(rs.context().query_devices())
        available_serials = []
        for device in devices:
            serial = safe_device_info(device, rs.camera_info.serial_number)
            available_serials.append(serial)
        self.selected_serial = choose_camera_serial(
            args.serial,
            available_serials,
        )

    def start(self) -> None:
        """启动指定数据流、占用会话目录并写入会话级元数据。"""

        rs = self.rs
        config = rs.config()
        config.enable_device(self.selected_serial)
        config.enable_stream(
            rs.stream.color,
            self.args.width,
            self.args.height,
            rs.format.bgr8,
            self.args.fps,
        )
        if not self.args.no_depth:
            config.enable_stream(
                rs.stream.depth,
                self.args.width,
                self.args.height,
                rs.format.z16,
                self.args.fps,
            )

        profile = self.pipeline.start(config)
        self.started = True
        reserve_session_directory(self.session_dir)
        if self.args.no_depth:
            self.align = None
        else:
            self.align = rs.align(rs.stream.color)

        device = profile.get_device()
        self.device_serial = safe_device_info(device, rs.camera_info.serial_number) or ""
        if self.device_serial != self.selected_serial:
            raise RuntimeError(
                "RealSense 实际启动设备与预选序列号不一致："
                f"selected={self.selected_serial!r}, actual={self.device_serial!r}"
            )
        self._write_session_metadata(profile, device)

    def _write_session_metadata(self, profile: Any, device: Any) -> None:
        """保存相机内参、流配置、设备身份和深度单位。"""

        rs = self.rs
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_profile = None
        depth_scale_m = None
        depth_to_color = None
        if not self.args.no_depth:
            depth_profile = profile.get_stream(
                rs.stream.depth
            ).as_video_stream_profile()
            depth_scale_m = device.first_depth_sensor().get_depth_scale()
            depth_to_color = extrinsics_to_dict(
                depth_profile.get_extrinsics_to(color_profile)
            )

        depth_intrinsics = None
        saved_depth_frame = None
        actual_depth_stream = None
        if depth_profile is not None:
            depth_intrinsics = intrinsics_to_dict(depth_profile.get_intrinsics())
            saved_depth_frame = "color_optical"
            actual_depth_stream = {
                "width": depth_profile.width(),
                "height": depth_profile.height(),
                "fps": depth_profile.fps(),
                "saved_aligned_to": "realsense_color_optical",
            }

        write_json_no_clobber(
            self.session_dir / "intrinsics.json",
            {
                "color": intrinsics_to_dict(color_profile.get_intrinsics()),
                "depth_native": depth_intrinsics,
                "depth_to_color": depth_to_color,
                "depth_scale_m": depth_scale_m,
                "saved_depth_is_aligned_to": saved_depth_frame,
            },
        )

        write_json_no_clobber(
            self.session_dir / "session.json",
            {
                "schema_version": 1,
                "created_at_local": datetime.now().astimezone().isoformat(),
                "camera": {
                    "name": safe_device_info(device, rs.camera_info.name),
                    "serial": self.device_serial,
                    "firmware_version": safe_device_info(
                        device, rs.camera_info.firmware_version
                    ),
                    "product_line": safe_device_info(
                        device, rs.camera_info.product_line
                    ),
                },
                "requested_stream": {
                    "width": self.args.width,
                    "height": self.args.height,
                    "fps": self.args.fps,
                    "color_format": "bgr8",
                    "depth_enabled": not self.args.no_depth,
                },
                "actual_color_stream": {
                    "width": color_profile.width(),
                    "height": color_profile.height(),
                    "fps": color_profile.fps(),
                    "frame": "realsense_color_optical",
                },
                "actual_depth_stream": actual_depth_stream,
            },
        )

    def read(self):
        """返回一组完整的彩色帧和可选对齐深度；不完整时返回 None。"""

        frames = self.pipeline.wait_for_frames()
        if self.align is not None:
            frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = None
        if not self.args.no_depth:
            depth_frame = frames.get_depth_frame()
        if not color_frame:
            return None
        if not self.args.no_depth and not depth_frame:
            print("warning: 当前帧缺少对齐深度，已跳过保存。")
            return None

        color = np.asanyarray(color_frame.get_data())
        depth = None
        if depth_frame:
            depth = np.asanyarray(depth_frame.get_data())
        return color_frame, depth_frame, color, depth

    def close(self) -> None:
        """释放当前对象占用的相机、文件或连接资源。"""

        if self.started:
            self.pipeline.stop()
        cv2.destroyAllWindows()


# =============================================================================
# 6. 实时采集循环
# =============================================================================

def capture_loop(camera: RealSenseCapture, args: argparse.Namespace) -> None:
    """显示实时彩色画面，并按键或定时保存完整帧。"""

    cv2.namedWindow("RealSense capture", cv2.WINDOW_NORMAL)
    print("Press s to save, q to quit.")
    print(f"Session: {camera.session_dir}")
    print(f"Camera serial: {camera.device_serial}")

    saved_count = 0
    last_auto_save = time.monotonic()
    while True:
        frame = camera.read()
        if frame is None:
            continue
        color_frame, depth_frame, color, depth = frame

        display = color.copy()
        cv2.putText(
            display,
            f"saved: {saved_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.imshow("RealSense capture", display)

        key = cv2.waitKey(1) & 0xFF
        now = time.monotonic()
        auto_due = (
            args.auto_interval > 0
            and now - last_auto_save >= args.auto_interval
        )
        if key == ord("q"):
            break
        if key == ord("s") or auto_due:
            depth_frame_number = None
            if depth_frame:
                depth_frame_number = depth_frame.get_frame_number()

            metadata = {
                "index": saved_count,
                "timestamp_unix_s": time.time(),
                "color_frame_number": color_frame.get_frame_number(),
                "depth_frame_number": depth_frame_number,
            }
            save_frame(camera.session_dir, saved_count, color, depth, metadata)
            saved_count += 1
            last_auto_save = now


# =============================================================================
# 7. 主流程
# =============================================================================

def main() -> int:
    """按照本脚本的编号流程依次执行各个步骤。"""

    args = parse_args()
    session_dir = validate_args(args)
    camera = RealSenseCapture(args, session_dir)
    try:
        camera.start()
        capture_loop(camera, args)
    finally:
        camera.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("\n用户取消采集。") from None
