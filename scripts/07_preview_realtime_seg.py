#!/usr/bin/env python3
"""第 07 步：使用 OpenCV 实时预览 YOLO26 实例分割结果。

这个脚本用于第 06 步训练完成后的第一轮视觉检查。当前模型应包含
``p-body``、``p-cap``、``y-body``、``y-cap`` 四类，它只负责：

- 从 RealSense D435/D435i、普通 USB 摄像头、视频或单张图片读取彩色画面；
- 使用训练得到的 YOLO Seg ``best.pt`` 做逐帧推理；
- 在 OpenCV 窗口中叠加分割掩膜、类别、置信度、实例数量和帧率；
- 可选保存截图、最后一帧可视化结果或整段预览视频。

脚本不会读取深度数据，也不会计算几何或控制机械臂。确认实时分割稳定以后，
当前主线建议继续使用脚本 15/17 做抓取前检查；历史几何检查脚本 08/09 已经
归档到 ``scripts_archive/``。若省略 ``--model``，脚本会递归搜索项目 ``runs``
目录，并选择修改时间最新的 ``best.pt``；这样既兼容新的标准训练目录，也兼容
早期的 ``runs/segment/runs/segment/...`` 嵌套目录。

RealSense 运行示例：

    python scripts/07_preview_realtime_seg.py \
      --source realsense \
      --serial REAL_SERIAL \
      --imgsz 1024 \
      --conf 0.25

如果需要指定某一次训练结果，可显式加入：

    --model runs/segment/tube_seg/weights/best.pt

窗口按键：

- ``Q`` 或 ``Esc``：退出；
- ``P`` 或空格：暂停/继续；
- ``S``：保存当前叠加画面；
- ``B``：显示/隐藏检测框；
- ``L``：显示/隐藏类别和置信度。
"""

# 启用较新的类型注解规则，让 ``str | None`` 等注解在运行时延迟求值。
from __future__ import annotations

# argparse 读取终端参数，例如 ``--source``、``--conf`` 和 ``--device``。
import argparse
# time.perf_counter() 提供高精度单调计时，用于计算实时总帧率。
import time
# Counter 用类别名称统计每一帧中各类实例的数量。
from collections import Counter
from collections.abc import Mapping, Sequence
# datetime 用当前时间生成不会轻易重复的截图文件名。
from datetime import datetime
# Path 负责模型、输入图片及输出文件的路径处理。
from pathlib import Path
# Any 表示不同帧源和 Ultralytics Result 共享的动态接口。
from typing import Any

# OpenCV 负责相机/视频读取、窗口显示、文字绘制和图片/视频保存。
import cv2
# NumPy 数组是 OpenCV 和 RealSense 彩色图像的共同数据格式。
import numpy as np


# 所有自动搜索和默认保存目录都相对于项目根目录，而不是终端当前目录。
ROOT = Path(__file__).resolve().parents[1]
# =============================================================================
# 本脚本自带的配置、模型类别和 RealSense 身份检查
# =============================================================================

"""当前四类试管实例分割模型的严格类别契约。

当前主线依赖相同的四类模型类别 ID：

    0 -> p-body
    1 -> p-cap
    2 -> y-body
    3 -> y-cap

只检查“名称集合中包含这些类别”是不够的。类别顺序改变会让后续显示、统计和
几何配对使用错误语义；多出其他类别也通常意味着自动选择到了别的实验权重。
因此这里直接在脚本内固定四类顺序，不再额外读取配置文件。
"""

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
) -> tuple[str, ...]:
    """严格校验模型任务和模型类别 ID 顺序。

    返回已验证的模型类别顺序，便于调用方打印或测试。函数只检查内存对象，
    不读取相机、不执行推理，也不会访问机械臂。
    """

    if str(task) != "segment":
        raise ValueError(f"模型任务必须是 segment，实际为 {task!r}。")

    actual = ordered_model_class_names(names)
    if actual != EXPECTED_TUBE_CLASS_ORDER:
        raise ValueError(
            "模型类别 ID/顺序必须与当前四类模型一致："
            f"实际={list(actual)}，"
            f"期望={list(EXPECTED_TUBE_CLASS_ORDER)}。"
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


# 这些扩展名会被 ``--source`` 识别为“只读取一次的单张图片”。
# 其他非数字 source 会交给 OpenCV，当作视频文件或视频地址尝试打开。
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class RealSenseColorSource:
    """RealSense 彩色帧源；本步骤不启用深度流或 IMU。"""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        serial: str | None,
    ) -> None:
        """按请求规格打开并验证唯一一台 RealSense 的彩色流。

        ``width``、``height`` 和 ``fps`` 是请求的流规格；设备最终采用的实际
        规格仍会从启动后的 profile 读取。``serial`` 是可选设备序列号：只接一台
        相机时可省略，检测到多台相机时必须明确指定。
        """

        # 延迟导入，使没有 RealSense SDK 的机器仍可查看 --help，或改用
        # USB 摄像头、视频和单张图片做预览。
        import pyrealsense2 as rs

        # 保存 SDK 对象，便于理解该实例的资源来自 pyrealsense2。
        self._rs = rs
        # pipeline 管理设备、流配置和逐帧读取。
        self._pipeline = rs.pipeline()
        # _started 记录 pipeline 是否成功启动，异常清理时据此决定能否 stop。
        self._started = False
        # 未指定 serial 且只有一台设备时自动选择；多台设备时会要求用户明确指定。
        selected_serial = select_realsense_device_serial(rs, serial)
        # config 描述“哪台设备、启用哪一种数据流”。
        config = rs.config()
        # 把配置绑定到已选择的序列号，避免 SDK 自动切换到另一台相机。
        config.enable_device(selected_serial)
        # 本脚本只启用 BGR8 彩色流，不读取或对齐深度流。
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        try:
            # start() 真正占用相机；返回 profile 可查询实际启用的设备和流参数。
            profile = self._pipeline.start(config)
            self._started = True
            device = profile.get_device()
            # 从实际启动设备再次读取序列号，防止预选与启动结果不一致。
            actual_serial = str(
                device.get_info(rs.camera_info.serial_number)
            )
            if actual_serial != selected_serial:
                raise RuntimeError(
                    "RealSense 实际启动设备与预选序列号不一致："
                    f"selected={selected_serial!r}, actual={actual_serial!r}"
                )
            # 获取实际彩色流 profile，而不是假设设备一定接受请求参数。
            color_profile = profile.get_stream(
                rs.stream.color
            ).as_video_stream_profile()
            # 对外暴露实际 FPS，供视频保存和 FPS 初值使用。
            self.fps = float(color_profile.fps())
            # description 会打印到终端，方便用户核对相机身份和流规格。
            self.description = (
                f"RealSense serial={actual_serial} color "
                f"{color_profile.width()}x{color_profile.height()}@"
                f"{color_profile.fps()}"
            )
        except Exception:
            # start 后任何校验失败都必须释放设备，再把原异常继续抛出。
            if self._started:
                self._pipeline.stop()
                self._started = False
            raise

    def read(self) -> np.ndarray | None:
        """等待下一张彩色帧并返回 BGR NumPy 数组。"""

        # 最多等待 5 秒；超时由 SDK 抛出异常，不会无限卡住。
        frames = self._pipeline.wait_for_frames(timeout_ms=5000)
        # 一个 frameset 可能含多种流，这里只取彩色帧。
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        # asanyarray 尽量零拷贝地把 SDK 缓冲区包装成 NumPy BGR 数组。
        return np.asanyarray(color_frame.get_data())

    def close(self) -> None:
        """停止 RealSense pipeline，释放相机。"""

        # close 可被异常清理安全调用多次：只有已启动时才 stop。
        if self._started:
            self._pipeline.stop()
            self._started = False


class OpenCVStreamSource:
    """用 ``cv2.VideoCapture`` 读取普通摄像头或视频文件。"""

    def __init__(
        self,
        source: int | str,
        requested_width: int,
        requested_height: int,
        requested_fps: int,
    ) -> None:
        """通过 OpenCV 打开摄像头编号、视频文件或网络流。

        ``source`` 为整数时代表摄像头编号，并尝试设置后三个请求规格；它为
        字符串时直接交给 ``cv2.VideoCapture``，视频自身的规格不会被主动修改。
        """

        # source 可以是整数摄像头编号，也可以是视频文件/网络流字符串。
        self._capture = cv2.VideoCapture(source)
        # 只有普通摄像头才主动请求宽、高和 FPS；视频文件保持自身属性。
        if isinstance(source, int):
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
            self._capture.set(cv2.CAP_PROP_FPS, requested_fps)

        # 打开失败时先 release，避免底层句柄残留。
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"OpenCV 无法打开视频源：{source}")

        # 某些摄像头/视频后端会返回 0 FPS，此时用用户请求值作为合理回退。
        detected_fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if detected_fps > 0.0:
            self.fps = detected_fps
        else:
            self.fps = float(requested_fps)
        self.description = f"OpenCV source {source}"

    def read(self) -> np.ndarray | None:
        """读取一帧；视频结束或读取失败时返回 ``None``。"""

        success, frame = self._capture.read()
        if success:
            return frame
        return None

    def close(self) -> None:
        """释放 OpenCV 的摄像头或视频句柄。"""

        self._capture.release()


class SingleImageSource:
    """只返回一次画面的单张图片帧源，适合无相机烟雾测试。"""

    def __init__(self, image_path: Path) -> None:
        """解码一张本地图片，并把它准备成只可读取一次的帧源。

        ``image_path`` 必须指向 OpenCV 能解码的图片；解码失败会立即抛错，
        而不会把坏图片静默当成“输入已经结束”。
        """

        # imread 默认返回 OpenCV 使用的 BGR 三通道图像。
        self._image = cv2.imread(str(image_path))
        if self._image is None:
            raise ValueError(f"图片无法解码：{image_path}")
        # 单图只应进入推理循环一次，避免 --max-frames=0 时无限重复。
        self._consumed = False
        # 单图没有真实帧率；1.0 仅作为统一帧源接口的占位值。
        self.fps = 1.0
        self.description = f"image {image_path}"

    def read(self) -> np.ndarray | None:
        """第一次返回图片副本，后续调用返回 ``None`` 表示输入结束。"""

        if self._consumed:
            return None
        self._consumed = True
        # 返回副本，避免后续绘制意外修改类中保存的原始图片。
        return self._image.copy()

    def close(self) -> None:
        """单张图片没有需要释放的外部资源。"""

        return None


def parse_args() -> argparse.Namespace:
    """定义并解析第 07 步实时分割预览参数。

    返回的 ``Namespace`` 同时包含输入源、模型推理、界面显示和文件保存四组
    参数。这里只定义类型和默认值，更严格的组合检查由 ``validate_args`` 完成。
    """

    # 使用模块文档第一行作为 ``--help`` 的简短说明，避免重复维护文案。
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # 模型可留空自动寻找 best.pt，也可以是本地文件或官方模型名称。
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "训练得到的 best.pt；省略时递归选择项目 runs 下修改时间最新的 "
            "best.pt。也可传入 Ultralytics 分割模型名。"
        ),
    )
    # source 统一接受 RealSense 关键字、摄像头编号、视频或单图路径。
    parser.add_argument(
        "--source",
        default="realsense",
        help=(
            "输入源：realsense、USB 摄像头编号（如 0）、视频路径或单张图片路径；"
            "默认 realsense。"
        ),
    )
    # 序列号只对 RealSense 有意义；多相机环境禁止靠枚举顺序猜设备。
    parser.add_argument(
        "--serial",
        default=None,
        help=(
            "仅在 --source realsense 时使用的设备序列号；检测到多台 "
            "RealSense 时必须显式指定。"
        ),
    )
    # width/height/fps 是请求相机流的规格，视频文件不会被强制改成这些值。
    parser.add_argument("--width", type=int, default=1280, help="相机宽度，默认 1280。")
    parser.add_argument("--height", type=int, default=720, help="相机高度，默认 720。")
    parser.add_argument("--fps", type=int, default=30, help="相机帧率，默认 30。")
    # imgsz 是模型内部缩放尺寸；越大通常边缘更细，但计算量也更高。
    parser.add_argument("--imgsz", type=int, default=1024, help="YOLO 推理尺寸，默认 1024。")
    # conf 越高，显示的低把握实例越少。
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="保留预测实例的最低置信度，范围 0..1，默认 0.25。",
    )
    # iou 控制 NMS 对重叠候选的抑制程度。
    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="NMS 的 IoU 阈值，范围 0..1，默认 0.70。",
    )
    # device="0" 表示第一块 CUDA GPU，也可显式写 cpu。
    parser.add_argument("--device", default="0", help="推理设备，默认使用 GPU 0。")
    # BooleanOptionalAction 会同时生成 --retina-masks 与 --no-retina-masks。
    parser.add_argument(
        "--retina-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="使用原图分辨率掩膜；默认开启，可用 --no-retina-masks 关闭。",
    )
    # max-det 防止单帧异常地产生过多候选，line-width 只控制绘图外观。
    parser.add_argument("--max-det", type=int, default=50, help="每帧最大实例数，默认 50。")
    parser.add_argument("--line-width", type=int, default=2, help="绘制线宽，默认 2。")
    # 0 表示不因帧数自动退出；测试单图时可设成很小的正数。
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="最多处理多少帧；0 表示持续运行，自动测试时可设为 1 或 3。",
    )
    # 无桌面服务器不能调用 imshow，此开关仍允许完整推理和文件保存。
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="不创建 OpenCV 窗口，用于自动化或无桌面环境测试。",
    )
    # save-video 每处理一帧就写一次；扩展名 .avi 选择 XVID，其他使用 mp4v。
    parser.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help="可选：把叠加结果保存为 MP4/AVI 视频。",
    )
    # save-vis 只在退出时写最后一张已经叠加掩膜和状态面板的画面。
    parser.add_argument(
        "--save-vis",
        type=Path,
        default=None,
        help="可选：退出时保存最后一帧叠加结果。",
    )
    # 启动参数检查时默认保护已有文件；用户显式确认 --overwrite 后才允许覆盖。
    # 这里不是像脚本 03 那样的原子 no-clobber 发布机制。
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的 --save-video/--save-vis；默认拒绝覆盖。",
    )
    # 交互窗口按 S 时，截图会以时间戳命名并保存到此目录。
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=ROOT / "runs" / "realtime" / "screenshots",
        help="按 S 保存截图的目录。",
    )
    # 读取 sys.argv、执行 int/float/Path 转换并返回 Namespace。
    return parser.parse_args()


def find_latest_best_weight() -> Path:
    """递归寻找项目 ``runs`` 下修改时间最新的训练最佳权重。

    递归搜索同时兼容当前的 ``runs/segment/<name>/weights/best.pt`` 和早期
    Ultralytics 产生的重复嵌套路径。
    """

    # rglob 会递归搜索所有层级；is_file 排除同名目录。
    candidates: list[Path] = []
    for path in (ROOT / "runs").rglob("best.pt"):
        if path.is_file():
            candidates.append(path)
    # 没有本地最佳权重时不擅自退回预训练模型，以免用户误以为在看训练成果。
    if not candidates:
        raise FileNotFoundError(
            "没有在 runs 下找到 best.pt。请通过 --model 指定训练权重。"
        )
    # 逐个比较修改时间，选出最新保存的 best.pt。
    latest_path = candidates[0]
    latest_time = latest_path.stat().st_mtime
    for path in candidates[1:]:
        modified_time = path.stat().st_mtime
        if modified_time > latest_time:
            latest_path = path
            latest_time = modified_time
    return latest_path.resolve()


def resolve_model_argument(value: str | None) -> str:
    """把 ``--model`` 解析为本地绝对路径或 Ultralytics 模型名。

    省略参数时优先使用最新训练好的 ``best.pt``；显式参数不是现有文件时原样
    交给 Ultralytics，因此仍支持 ``yolo26x-seg.pt`` 这类官方模型名称。
    """

    # None 表示用户完全省略 --model，此时启用自动搜索。
    if value is None:
        return str(find_latest_best_weight())

    # expanduser() 支持 ``~/...``；若确实是文件就返回规范化绝对路径。
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    # 允许 yolo26x-seg.pt 这类 Ultralytics 可自动下载的模型名称。
    return value


def open_frame_source(args: argparse.Namespace) -> Any:
    """根据 ``--source`` 创建 RealSense、摄像头、视频或图片帧源。"""

    # argparse 的 source 默认是字符串；显式转成字符串并去掉意外空白。
    source_text = str(args.source).strip()
    # 关键字大小写不敏感，例如 RealSense 与 realsense 等价。
    if source_text.lower() == "realsense":
        return RealSenseColorSource(
            args.width,
            args.height,
            args.fps,
            args.serial,
        )
    # 非 RealSense 输入不应携带序列号，否则很可能是用户写错了 source。
    if args.serial is not None:
        raise ValueError("--serial 只能与 --source realsense 一起使用。")

    # 只有实际存在且扩展名在白名单中的文件才按单张图片处理。
    source_path = Path(source_text).expanduser()
    if source_path.is_file() and source_path.suffix.lower() in IMAGE_SUFFIXES:
        return SingleImageSource(source_path.resolve())

    # 全数字字符串解释为摄像头编号；其他内容交给 VideoCapture，因而可处理
    # 本地视频路径以及 OpenCV 支持的网络视频地址。
    if source_text.isdigit():
        return OpenCVStreamSource(
            int(source_text),
            args.width,
            args.height,
            args.fps,
        )
    return OpenCVStreamSource(
        source_text,
        args.width,
        args.height,
        args.fps,
    )


def class_count_text(result: Any) -> str:
    """按类别统计当前帧实例数，例如 ``p-body:1 | y-cap:1``。"""

    # 没有检测框就没有实例；返回固定文本避免后面访问空张量。
    if result.boxes is None or len(result.boxes) == 0:
        return "objects: none"

    # Ultralytics 的 cls 通常是 GPU tensor；先搬到 CPU，再转成整数数组。
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    # 用模型 names 字典把类别 ID 映射成可读名称，并统计出现次数。
    detected_names: list[str] = []
    for class_id in class_ids:
        detected_names.append(result.names[int(class_id)])
    counts = Counter(detected_names)
    # 名称排序后输出，使面板每帧的类别顺序稳定，不随检测顺序跳动。
    count_parts: list[str] = []
    for name in sorted(counts):
        count_parts.append(f"{name}:{counts[name]}")
    return " | ".join(count_parts)


def draw_status_panel(
    image: np.ndarray,
    fps: float,
    result: Any,
    model_name: str,
    show_boxes: bool,
    show_labels: bool,
    paused: bool,
) -> np.ndarray:
    """在分割叠加画面的左上角绘制模型、帧率、耗时和四类实例统计。"""

    # speed 是 Ultralytics 结果中的耗时字典；字段缺失时安全回退为 0。
    inference_ms = float(result.speed.get("inference", 0.0))
    # 每个字符串对应状态面板中的一行。
    boxes_text = "off"
    if show_boxes:
        boxes_text = "on"
    labels_text = "off"
    if show_labels:
        labels_text = "on"
    paused_text = ""
    if paused:
        paused_text = " | PAUSED"

    lines = [
        f"model: {model_name}",
        f"FPS: {fps:.1f} | inference: {inference_ms:.1f} ms",
        class_count_text(result),
        f"boxes:{boxes_text} labels:{labels_text}{paused_text}",
        "Q/ESC quit | P/SPACE pause | S snapshot | B boxes | L labels",
    ]

    # 面板高度随行数变化；宽度最多 720，但不会超过实际图像宽度。
    line_height = 25
    panel_height = 14 + line_height * len(lines)
    panel_width = min(image.shape[1], 720)
    # 在副本上画纯黑矩形，再与原图混合成半透明背景。
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0.0, image)

    # index 是列表下标；用它计算每一行文字的纵坐标。
    for index in range(len(lines)):
        text = lines[index]
        cv2.putText(
            image,
            text,
            (12, 25 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    # OpenCV 绘图会原地修改 image；返回它便于调用者继续传递。
    return image


def create_video_writer(
    output_path: Path,
    frame_shape: tuple[int, ...],
    fps: float,
) -> cv2.VideoWriter:
    """根据第一帧的实际尺寸创建可视化视频输出器。"""

    # 固定为绝对路径，并自动创建缺失的父目录。
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # OpenCV frame_shape 通常为 (height, width, channels)。
    height, width = frame_shape[:2]
    # AVI 使用 XVID；MP4 及其他扩展名使用常见的 mp4v 编码。
    if output_path.suffix.lower() == ".avi":
        codec = "XVID"
    else:
        codec = "mp4v"
    # max(fps, 1.0) 防止某些输入源报告 0 FPS 导致 VideoWriter 创建失败。
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(codec[0], codec[1], codec[2], codec[3]),
        max(fps, 1.0),
        (width, height),
    )
    # 构造对象并不代表编码器真的可用，必须用 isOpened 再次确认。
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"无法创建输出视频：{output_path}")
    return writer


def save_snapshot(image: np.ndarray, output_dir: Path) -> Path:
    """保存当前可视化画面，并返回文件路径。"""

    # 截图目录不存在时递归创建。
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # 微秒级时间戳降低连续按键时文件名碰撞的概率。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"seg_{timestamp}.jpg"
    # imwrite 返回布尔值，False 表示编码或写盘失败，不能当作保存成功。
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"截图保存失败：{output_path}")
    return output_path


def validate_args(args: argparse.Namespace) -> None:
    """在加载模型和相机前拒绝明显错误的参数。"""

    # 相机流尺寸和帧率必须是正数，即使当前输入可能是视频或图片。
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise ValueError("--width、--height 和 --fps 必须大于 0。")
    # 模型尺寸、最大实例数和绘制线宽同样不接受 0 或负数。
    if args.imgsz <= 0 or args.max_det <= 0 or args.line_width <= 0:
        raise ValueError("--imgsz、--max-det 和 --line-width 必须大于 0。")
    # 置信度和 IoU 都是比例，合法闭区间为 0..1。
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        raise ValueError("--conf 和 --iou 必须位于 0..1。")
    # 0 有特殊含义“无限运行”，所以只拒绝负数。
    if args.max_frames < 0:
        raise ValueError("--max-frames 不能小于 0。")
    # 过滤 None 后，把两个可选输出都规范化成绝对路径。
    outputs: list[Path] = []
    for path in (args.save_video, args.save_vis):
        if path is not None:
            outputs.append(path.expanduser().resolve())
    # 同一文件不能既被当视频又被当图片写入。
    if len(outputs) != len(set(outputs)):
        raise ValueError("--save-video 与 --save-vis 不能指向同一个文件。")
    # 未获得 --overwrite 明确授权时，启动模型前就拒绝现有目标文件。
    if not args.overwrite:
        existing_outputs: list[Path] = []
        for path in outputs:
            if path.exists():
                existing_outputs.append(path)
        if existing_outputs:
            raise FileExistsError(
                "输出文件已存在，默认拒绝覆盖；请更换路径或显式添加 "
                f"--overwrite：{existing_outputs}"
            )
    # 若输入是本地文件，还要阻止输出路径覆盖正在读取的输入本身。
    source_path = Path(str(args.source)).expanduser()
    if source_path.is_file():
        resolved_source = source_path.resolve()
        if resolved_source in outputs:
            raise ValueError("输出文件不能与 --source 输入文件相同。")


def main() -> int:
    """加载模型、打开帧源并运行实时 OpenCV 预览循环。"""

    # 第一步：读取参数并在占用 GPU/相机之前完成低成本校验。
    args = parse_args()
    validate_args(args)

    # 延迟导入，保证即使没有安装 Ultralytics 也能先查看 --help。
    from ultralytics import YOLO

    # 未传 --model 时自动选最新 best.pt，并把最终选择打印出来，便于确认实时
    # 画面使用的是哪一次训练结果。
    model_argument = resolve_model_argument(args.model)
    print(f"Model: {model_argument}")
    # YOLO(...) 会读取本地权重；官方模型名不存在时可能触发 Ultralytics 下载。
    model = YOLO(model_argument)
    # 在 open_frame_source() 打开相机/视频，以及任何 model.predict() 之前，
    # 严格检查 task、类别数量和类别 ID 顺序。
    validate_tube_model_contract(
        task=model.task,
        names=model.names,
    )
    print(f"Classes: {model.names}")

    # 类别契约通过后才打开相机或视频，避免错误模型占用相机。
    source = open_frame_source(args)
    print(f"Source: {source.description}")
    print("Keys: Q/ESC quit, P/SPACE pause, S snapshot, B boxes, L labels")

    # writer 延迟到得到第一帧实际尺寸后才创建；未要求保存视频时始终为 None。
    writer: cv2.VideoWriter | None = None
    # 保存最近一张叠加完成的画面，供暂停显示、截图和退出保存使用。
    last_visualization: np.ndarray | None = None
    # 下面三个布尔量分别记录框、标签和暂停状态，可由按键切换。
    show_boxes = True
    show_labels = True
    paused = False
    # 第一帧额外做一次预测作为 GPU/模型预热，但不计为 processed_frames。
    warmed_up = False
    # 只统计真正完成推理和绘制的帧，用于 --max-frames 自动退出。
    processed_frames = 0
    # FPS 使用相邻两张“完成绘制的画面”之间的时间计算，包含相机取帧、
    # 模型推理和 OpenCV 绘制，而不是只显示 GPU 推理速度。
    fps_ema = float(source.fps)
    last_frame_completed_at: float | None = None
    # 固定窗口名也用于 destroyAllWindows 前的人类识别。
    window_name = "HPS realtime segmentation"

    # 无桌面模式完全不创建 GUI 窗口。
    if not args.no_display:
        try:
            # WINDOW_NORMAL 允许用户拖动调整窗口大小。
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except Exception:
            # 窗口创建失败也要及时释放已经打开的相机。
            source.close()
            raise

    try:
        # 主循环持续到输入结束、按退出键、达到最大帧数或发生异常。
        while True:
            # 暂停时保留上一画面，不从相机取新帧，也不调用模型。
            if not paused:
                frame = source.read()
                # 视频结束、单图已消费或帧源返回失败时正常退出循环。
                if frame is None:
                    break

                # 第一帧先完成模型/GPU 预热，后续显示的 FPS 更接近稳定速度。
                if not warmed_up:
                    model.predict(
                        source=frame,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        iou=args.iou,
                        device=args.device,
                        # 项目视觉脚本统一固定使用 FP32。
                        half=False,
                        max_det=args.max_det,
                        retina_masks=args.retina_masks,
                        verbose=False,
                    )
                    warmed_up = True

                # predict 返回结果列表；单帧输入只取索引 0。
                prediction_results = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    half=False,
                    max_det=args.max_det,
                    retina_masks=args.retina_masks,
                    verbose=False,
                )
                result = prediction_results[0]

                # color_mode="class" 让同一类别始终使用一致颜色，便于观察四类
                # p-body、p-cap、y-body、y-cap 是否稳定。
                visualization = result.plot(
                    # 在标签中显示置信度，并按交互状态决定框和文字。
                    conf=True,
                    line_width=args.line_width,
                    labels=show_labels,
                    boxes=show_boxes,
                    # 始终绘制实例分割掩膜。
                    masks=True,
                    color_mode="class",
                )

                # 记录整帧完成时间，并由相邻完成时刻计算“端到端 FPS”。
                frame_completed_at = time.perf_counter()
                if last_frame_completed_at is not None:
                    # 1e-9 防止极端计时精度下出现除零。
                    elapsed = max(frame_completed_at - last_frame_completed_at, 1e-9)
                    instant_fps = 1.0 / elapsed
                    # 指数移动平均抑制单帧波动：88% 历史值 + 12% 当前值。
                    fps_ema = 0.88 * fps_ema + 0.12 * instant_fps
                last_frame_completed_at = frame_completed_at

                # 把状态面板写到当前叠加画面，并记为“最近一帧”。
                last_visualization = draw_status_panel(
                    visualization,
                    fps_ema,
                    result,
                    Path(model_argument).name,
                    show_boxes,
                    show_labels,
                    paused=False,
                )
                processed_frames += 1

                # 只有用户指定 --save-video 时才创建和写入 VideoWriter。
                if args.save_video is not None:
                    if writer is None:
                        # 首帧确定输出宽高；帧源 FPS 决定视频播放速度。
                        writer = create_video_writer(
                            args.save_video,
                            last_visualization.shape,
                            source.fps,
                        )
                    writer.write(last_visualization)

            # 防御性保护：正常路径中成功读取和推理后应已有画面；若未来帧源
            # 实现允许“本轮无结果但输入未结束”，这里就跳过显示和按键处理。
            if last_visualization is None:
                continue

            # GUI 模式显示画面并读取按键；无桌面模式使用 255 表示“无按键”。
            if not args.no_display:
                shown = last_visualization
                if paused:
                    # 暂停面板画在副本上，避免把 PAUSED 永久写进保存画面。
                    shown = draw_status_panel(
                        last_visualization.copy(),
                        fps_ema,
                        result,
                        Path(model_argument).name,
                        show_boxes,
                        show_labels,
                        paused=True,
                    )
                cv2.imshow(window_name, shown)
                # 暂停时等待 30 ms，正常时只等 1 ms，& 0xFF 统一不同平台键码。
                wait_milliseconds = 1
                if paused:
                    wait_milliseconds = 30
                key = cv2.waitKey(wait_milliseconds) & 0xFF
            else:
                key = 255

            # Esc 的键码是 27；同时接受大小写 Q。
            if key in (27, ord("q"), ord("Q")):
                break
            # P 或空格切换暂停；elif 保证一次按键只触发一个动作。
            if key in (ord("p"), ord("P"), ord(" ")):
                paused = not paused
            elif key in (ord("s"), ord("S")):
                # 截图保存当前未带 PAUSED 临时字样的叠加结果。
                snapshot_path = save_snapshot(last_visualization, args.snapshot_dir)
                print(f"Snapshot: {snapshot_path}")
            elif key in (ord("b"), ord("B")):
                show_boxes = not show_boxes
            elif key in (ord("l"), ord("L")):
                show_labels = not show_labels

            # 正数限制达到后退出；0 表示不按帧数结束。
            if args.max_frames > 0 and processed_frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        # Ctrl+C 是用户主动停止，打印简短提示后仍进入 finally 清理资源。
        print("Stopped by Ctrl+C.")
    finally:
        # 无论正常退出还是出现异常，都必须释放相机/视频和输出文件。
        try:
            source.close()
        finally:
            if writer is not None:
                writer.release()
            if not args.no_display:
                cv2.destroyAllWindows()

    # --save-vis 在循环完全结束后保存最后一帧；没有成功帧时不生成空文件。
    if args.save_vis is not None and last_visualization is not None:
        save_path = args.save_vis.expanduser().resolve()
        # 长时间运行期间目标文件可能被其他进程新建，因此写入前再次检查。
        if save_path.exists() and not args.overwrite:
            raise FileExistsError(
                "--save-vis 在运行期间已出现，默认拒绝覆盖；"
                f"请更换路径或显式添加 --overwrite：{save_path}"
            )
        # 创建父目录后调用 imwrite，并检查它返回的成功标志。
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(save_path), last_visualization):
            raise RuntimeError(f"最后一帧保存失败：{save_path}")
        print(f"Visualization: {save_path}")

    # 最后打印实际处理量和指数平滑 FPS，便于无窗口运行时核对结果。
    print(f"Processed frames: {processed_frames}")
    if fps_ema > 0.0:
        print(f"Smoothed FPS: {fps_ema:.2f}")
    # 0 表示脚本正常完成。
    return 0


# 直接运行脚本时才进入 main；测试代码 import 本文件不会打开模型或相机。
if __name__ == "__main__":
    # SystemExit 把 main() 返回的 0 作为终端进程退出码。
    raise SystemExit(main())
