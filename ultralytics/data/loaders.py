# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import glob
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import torch
from PIL import Image

from ultralytics.data.utils import FORMATS_HELP_MSG, IMG_FORMATS, VID_FORMATS
from ultralytics.utils import IS_COLAB, IS_KAGGLE, LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.patches import imread


@dataclass
class SourceTypes:
    """
    用于表示不同类型输入源的类，用于进行预测。

    该类使用 dataclass 来定义布尔标志，表示可以用于使用 YOLO 模型进行预测的不同类型的输入源。

    属性：
        stream (bool): 标志，指示输入源是否是视频流。
        screenshot (bool): 标志，指示输入源是否是截图。
        from_img (bool): 标志，指示输入源是否是图像文件。

    示例：
        >>> source_types = SourceTypes(stream=True, screenshot=False, from_img=False)
        >>> print(source_types.stream)
        True
        >>> print(source_types.from_img)
        False
    """

    stream: bool = False
    screenshot: bool = False
    from_img: bool = False
    tensor: bool = False


class LoadStreams:
    """
    用于加载各种类型视频流的流加载器。

    支持RTSP、RTMP、HTTP和TCP视频流。该类处理多个视频流的加载和处理，使其适用于实时视频分析任务。

    属性：
        sources (List[str]): 视频流的源路径或URL列表。
        vid_stride (int): 视频帧率步长。
        buffer (bool): 是否缓冲输入流。
        running (bool): 标志位，指示流线程是否正在运行。
        mode (str): 设置为“stream”，表示实时捕捉。
        imgs (List[List[np.ndarray]]): 每个视频流的图像帧列表。
        fps (List[float]): 每个视频流的帧率列表。
        frames (List[int]): 每个视频流的总帧数列表。
        threads (List[Thread]): 每个视频流的线程列表。
        shape (List[Tuple[int, int, int]]): 每个视频流的图像形状列表。
        caps (List[cv2.VideoCapture]): 每个视频流的cv2.VideoCapture对象列表。
        bs (int): 处理的批次大小。

    方法：
        update: 在守护线程中读取流帧。
        close: 关闭流加载器并释放资源。
        __iter__: 返回类的迭代器对象。
        __next__: 返回源路径、转换后的图像和原始图像以供处理。
        __len__: 返回sources对象的长度。

    示例：
        >>> stream_loader = LoadStreams("rtsp://example.com/stream1.mp4")
        >>> for sources, imgs, _ in stream_loader:
        ...     # 处理图像
        ...     pass
        >>> stream_loader.close()

    注意：
        - 该类使用多线程高效地从多个流中同时加载帧。
        - 它自动处理YouTube链接，将其转换为最佳可用的流URL。
        - 该类实现了一个缓冲系统，用于管理帧的存储和检索。
    """

    def __init__(self, sources="file.streams", vid_stride=1, buffer=False):
        """初始化用于多个视频源的流加载器，支持各种流类型。"""
        torch.backends.cudnn.benchmark = True  # 对于固定大小的推理更快
        self.buffer = buffer  # 缓冲输入流
        self.running = True  # 线程运行标志
        self.mode = "stream"
        self.vid_stride = vid_stride  # 视频帧率步长

        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.bs = n
        self.fps = [0] * n  # 帧率
        self.frames = [0] * n
        self.threads = [None] * n
        self.caps = [None] * n  # 视频捕捉对象
        self.imgs = [[] for _ in range(n)]  # 图像
        self.shape = [[] for _ in range(n)]  # 图像形状
        self.sources = [ops.clean_str(x) for x in sources]  # 清理源名称，供后续使用
        for i, s in enumerate(sources):  # 索引，源
            # 启动线程以从视频流中读取帧
            st = f"{i + 1}/{n}: {s}... "
            if urlparse(s).hostname in {"www.youtube.com", "youtube.com", "youtu.be"}:  # 如果源是YouTube视频
                # YouTube格式，如 'https://www.youtube.com/watch?v=Jsn8D3aC840' 或 'https://youtu.be/Jsn8D3aC840'
                s = get_best_youtube_url(s)
            s = eval(s) if s.isnumeric() else s  # 即：s = '0' 本地摄像头
            if s == 0 and (IS_COLAB or IS_KAGGLE):
                raise NotImplementedError(
                    "'source=0' 摄像头在Colab和Kaggle笔记本中不受支持。"
                    " 请在本地环境中运行 'source=0'。"
                )
            self.caps[i] = cv2.VideoCapture(s)  # 存储视频捕捉对象
            if not self.caps[i].isOpened():
                raise ConnectionError(f"{st}无法打开 {s}")
            w = int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.caps[i].get(cv2.CAP_PROP_FPS)  # 警告：可能返回0或nan
            self.frames[i] = max(int(self.caps[i].get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float(
                "inf"
            )  # 无限流回退
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS回退

            success, im = self.caps[i].read()  # 确保读取第一帧
            if not success or im is None:
                raise ConnectionError(f"{st}无法从 {s} 读取图像")
            self.imgs[i].append(im)
            self.shape[i] = im.shape
            self.threads[i] = Thread(target=self.update, args=([i, self.caps[i], s]), daemon=True)
            LOGGER.info(f"{st}成功 ✅ ({self.frames[i]} 帧，形状 {w}x{h}，帧率 {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info("")  # 换行

    def update(self, i, cap, stream):
        """在守护线程中读取流帧并更新图像缓冲区。"""
        n, f = 0, self.frames[i]  # 帧号，帧数组
        while self.running and cap.isOpened() and n < (f - 1):
            if len(self.imgs[i]) < 30:  # 保持缓冲区中的图像不超过30张
                n += 1
                cap.grab()  # .read() = .grab() 然后是 .retrieve()
                if n % self.vid_stride == 0:
                    success, im = cap.retrieve()
                    if not success:
                        im = np.zeros(self.shape[i], dtype=np.uint8)
                        LOGGER.warning("警告 ⚠️ 视频流无响应，请检查您的IP摄像头连接。")
                        cap.open(stream)  # 如果信号丢失，重新打开流
                    if self.buffer:
                        self.imgs[i].append(im)
                    else:
                        self.imgs[i] = [im]
            else:
                time.sleep(0.01)  # 等待直到缓冲区为空

    def close(self):
        """终止流加载器，停止线程并释放视频捕捉资源。"""
        self.running = False  # 线程停止标志
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)  # 添加超时
        for cap in self.caps:  # 遍历存储的VideoCapture对象
            try:
                cap.release()  # 释放视频捕捉
            except Exception as e:
                LOGGER.warning(f"警告 ⚠️ 无法释放 VideoCapture 对象: {e}")
        cv2.destroyAllWindows()

    def __iter__(self):
        """迭代YOLO图像流，并重新打开无响应的流。"""
        self.count = -1
        return self

    def __next__(self):
        """返回多个视频流的下一个批次帧用于处理。"""
        self.count += 1

        images = []
        for i, x in enumerate(self.imgs):
            # 等待每个缓冲区中有一帧可用
            while not x:
                if not self.threads[i].is_alive() or cv2.waitKey(1) == ord("q"):  # 按q退出
                    self.close()
                    raise StopIteration
                time.sleep(1 / min(self.fps))
                x = self.imgs[i]
                if not x:
                    LOGGER.warning(f"警告 ⚠️ 正在等待流 {i}")

            # 获取并移除imgs缓冲区中的第一帧
            if self.buffer:
                images.append(x.pop(0))

            # 获取最后一帧，并清除imgs缓冲区中的其他帧
            else:
                images.append(x.pop(-1) if x else np.zeros(self.shape[i], dtype=np.uint8))
                x.clear()

        return self.sources, images, [""] * self.bs

    def __len__(self):
        """返回LoadStreams对象中的视频流数量。"""
        return self.bs  # 1E12帧 = 32个流，每秒30帧，持续30年


class LoadScreenshots:
    """
    Ultralytics 截图数据加载器，用于捕获和处理屏幕图像。

    该类管理截图图像的加载，用于与 YOLO 一起处理。适用于使用 `yolo predict source=screen`。

    属性：
        source (str): 输入源，指示要捕获的屏幕。
        screen (int): 要捕获的屏幕编号。
        left (int): 屏幕捕获区域的左侧坐标。
        top (int): 屏幕捕获区域的顶部坐标。
        width (int): 屏幕捕获区域的宽度。
        height (int): 屏幕捕获区域的高度。
        mode (str): 设置为 'stream'，表示实时捕获。
        frame (int): 捕获的帧计数器。
        sct (mss.mss): 来自 `mss` 库的屏幕捕获对象。
        bs (int): 批处理大小，设置为 1。
        fps (int): 每秒帧数，设置为 30。
        monitor (Dict[str, int]): 显示器的配置详细信息。

    方法：
        __iter__: 返回一个迭代器对象。
        __next__: 捕获下一个截图并返回它。

    示例：
        >>> loader = LoadScreenshots("0 100 100 640 480")  # 屏幕 0，左上角 (100,100)，640x480
        >>> for source, im, im0s, vid_cap, s in loader:
        ...     print(f"捕获的帧: {im.shape}")
    """

    def __init__(self, source):
        """根据指定的屏幕和区域参数初始化截图捕获。"""
        check_requirements("mss")
        import mss  # noqa

        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None  # 默认捕获整个屏幕 0
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.mode = "stream"
        self.frame = 0
        self.sct = mss.mss()
        self.bs = 1
        self.fps = 30

        # 解析显示器形状
        monitor = self.sct.monitors[self.screen]
        self.top = monitor["top"] if top is None else (monitor["top"] + top)
        self.left = monitor["left"] if left is None else (monitor["left"] + left)
        self.width = width or monitor["width"]
        self.height = height or monitor["height"]
        self.monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def __iter__(self):
        """从指定的屏幕或区域中获取下一个截图图像供处理。"""
        return self

    def __next__(self):
        """使用 mss 库捕获并返回下一个截图作为 numpy 数组。"""
        im0 = np.asarray(self.sct.grab(self.monitor))[:, :, :3]  # 从 BGRA 转换为 BGR
        s = f"屏幕 {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: "

        self.frame += 1
        return [str(self.screen)], [im0], [s]  # 屏幕、图像、字符串


class LoadImagesAndVideos:
    """
    用于加载和处理YOLO目标检测的图像和视频的类。

    该类管理来自不同来源的图像和视频数据的加载和预处理，包括单个图像文件、视频文件和图像及视频路径的列表。

    属性：
        files (List[str]): 图像和视频文件路径列表。
        nf (int): 文件的总数（图像和视频）。
        video_flag (List[bool]): 标志位，指示文件是视频（True）还是图像（False）。
        mode (str): 当前模式，“image”或“video”。
        vid_stride (int): 视频帧率的步长。
        bs (int): 批次大小。
        cap (cv2.VideoCapture): OpenCV的视频捕捉对象。
        frame (int): 视频的帧计数器。
        frames (int): 视频的总帧数。
        count (int): 迭代计数器，在__iter__()时初始化为0。
        ni (int): 图像数量。

    方法：
        __init__: 初始化LoadImagesAndVideos对象。
        __iter__: 返回VideoStream或ImageFolder的迭代器对象。
        __next__: 返回下一批图像或视频帧以及它们的路径和元数据。
        _new_video: 为给定路径创建一个新的视频捕捉对象。
        __len__: 返回对象中的批次数。

    示例：
        >>> loader = LoadImagesAndVideos("path/to/data", batch=32, vid_stride=1)
        >>> for paths, imgs, info in loader:
        ...     # 处理图像或视频帧的批次
        ...     pass

    注意：
        - 支持多种图像格式，包括HEIC。
        - 处理本地文件和目录。
        - 可以从包含图像和视频路径的文本文件中读取。
    """

    def __init__(self, path, batch=1, vid_stride=1):
        """初始化用于图像和视频的数据加载器，支持各种输入格式。"""
        parent = None
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt文件，每行包含img/vid/dir
            parent = Path(path).parent
            path = Path(path).read_text().splitlines()  # 源列表
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            a = str(Path(p).absolute())  # 不使用.resolve() https://github.com/ultralytics/ultralytics/issues/2912
            if "*" in a:
                files.extend(sorted(glob.glob(a, recursive=True)))  # glob
            elif os.path.isdir(a):
                files.extend(sorted(glob.glob(os.path.join(a, "*.*"))))  # 目录
            elif os.path.isfile(a):
                files.append(a)  # 文件（绝对路径或相对于CWD）
            elif parent and (parent / p).is_file():
                files.append(str((parent / p).absolute()))  # 文件（相对于*.txt文件的父目录）
            else:
                raise FileNotFoundError(f"{p} 不存在")

        # 定义文件为图像或视频
        images, videos = [], []
        for f in files:
            suffix = f.split(".")[-1].lower()  # 获取文件扩展名（去掉点并转为小写）
            if suffix in IMG_FORMATS:
                images.append(f)
            elif suffix in VID_FORMATS:
                videos.append(f)
        ni, nv = len(images), len(videos)

        self.files = images + videos
        self.nf = ni + nv  # 文件数量
        self.ni = ni  # 图像数量
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "video" if ni == 0 else "image"  # 如果没有图像，默认设置为视频
        self.vid_stride = vid_stride  # 视频帧率步长
        self.bs = batch
        if any(videos):
            self._new_video(videos[0])  # 新的视频
        else:
            self.cap = None
        if self.nf == 0:
            raise FileNotFoundError(f"在 {p} 中没有找到图像或视频。 {FORMATS_HELP_MSG}")

    def __iter__(self):
        """迭代图像/视频文件，返回源路径、图像和元数据。"""
        self.count = 0
        return self

    def __next__(self):
        """返回下一批图像或视频帧以及它们的路径和元数据。"""
        paths, imgs, info = [], [], []
        while len(imgs) < self.bs:
            if self.count >= self.nf:  # 文件列表结束
                if imgs:
                    return paths, imgs, info  # 返回最后一个部分批次
                else:
                    raise StopIteration

            path = self.files[self.count]
            if self.video_flag[self.count]:
                self.mode = "video"
                if not self.cap or not self.cap.isOpened():
                    self._new_video(path)

                success = False
                for _ in range(self.vid_stride):
                    success = self.cap.grab()
                    if not success:
                        break  # 视频结束或失败

                if success:
                    success, im0 = self.cap.retrieve()
                    if success:
                        self.frame += 1
                        paths.append(path)
                        imgs.append(im0)
                        info.append(f"视频 {self.count + 1}/{self.nf} (帧 {self.frame}/{self.frames}) {path}: ")
                        if self.frame == self.frames:  # 视频结束
                            self.count += 1
                            self.cap.release()
                else:
                    # 如果当前视频结束或打开失败，跳转到下一个文件
                    self.count += 1
                    if self.cap:
                        self.cap.release()
                    if self.count < self.nf:
                        self._new_video(self.files[self.count])
            else:
                # 处理图像文件（包括HEIC）
                self.mode = "image"
                if path.split(".")[-1].lower() == "heic":
                    # 使用Pillow和pillow-heif加载HEIC图像
                    check_requirements("pillow-heif")

                    from pillow_heif import register_heif_opener

                    register_heif_opener()  # 使用Pillow注册HEIF解码器
                    with Image.open(path) as img:
                        im0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)  # 转换图像为BGR的NumPy数组
                else:
                    im0 = imread(path)  # BGR
                if im0 is None:
                    LOGGER.warning(f"警告 ⚠️ 图像读取错误 {path}")
                else:
                    paths.append(path)
                    imgs.append(im0)
                    info.append(f"图像 {self.count + 1}/{self.nf} {path}: ")
                self.count += 1  # 移动到下一个文件
                if self.count >= self.ni:  # 图像列表结束
                    break

        return paths, imgs, info

    def _new_video(self, path):
        """为给定路径创建新的视频捕捉对象并初始化与视频相关的属性。"""
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"无法打开视频 {path}")
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)

    def __len__(self):
        """返回数据集中的文件数量（图像和视频）。"""
        return math.ceil(self.nf / self.bs)  # 批次数


class LoadPilAndNumpy:
    """
    从PIL和Numpy数组加载图像以进行批处理。

    该类管理从PIL和Numpy格式加载和预处理图像数据。它执行基本的验证和格式转换，以确保图像符合下游处理所需的格式。

    属性：
        paths (List[str]): 图像路径或自动生成的文件名列表。
        im0 (List[np.ndarray]): 存储为Numpy数组的图像列表。
        mode (str): 处理的数据类型，设置为'image'。
        bs (int): 批次大小，相当于`im0`的长度。

    方法：
        _single_check: 验证并格式化单个图像为Numpy数组。

    示例：
        >>> from PIL import Image
        >>> import numpy as np
        >>> pil_img = Image.new("RGB", (100, 100))
        >>> np_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> loader = LoadPilAndNumpy([pil_img, np_img])
        >>> paths, images, _ = next(iter(loader))
        >>> print(f"Loaded {len(images)} images")
        Loaded 2 images
    """

    def __init__(self, im0):
        """初始化PIL和Numpy图像加载器，将输入转换为标准化格式。"""
        if not isinstance(im0, list):
            im0 = [im0]
        # 当Image.filename返回空路径时，使用`image{i}.jpg`。
        self.paths = [getattr(im, "filename", "") or f"image{i}.jpg" for i, im in enumerate(im0)]
        self.im0 = [self._single_check(im) for im in im0]
        self.mode = "image"
        self.bs = len(self.im0)

    @staticmethod
    def _single_check(im):
        """验证并将图像格式化为Numpy数组，确保RGB顺序和连续内存。"""
        assert isinstance(im, (Image.Image, np.ndarray)), f"期望PIL/np.ndarray图像类型，但得到{type(im)}"
        if isinstance(im, Image.Image):
            if im.mode != "RGB":
                im = im.convert("RGB")
            im = np.asarray(im)[:, :, ::-1]
            im = np.ascontiguousarray(im)  # 连续内存
        return im

    def __len__(self):
        """返回'im0'属性的长度，表示已加载的图像数量。"""
        return len(self.im0)

    def __next__(self):
        """返回下一批图像、路径和元数据以供处理。"""
        if self.count == 1:  # 只循环一次，因为是批量推理
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, [""] * self.bs

    def __iter__(self):
        """迭代PIL/Numpy图像，返回路径、原始图像和元数据以供处理。"""
        self.count = 0
        return self


class LoadTensor:
    """
    用于加载和处理张量数据的类，适用于目标检测任务。

    该类处理从 PyTorch 张量加载和预处理图像数据，为进一步的目标检测流水线处理做好准备。

    属性：
        im0 (torch.Tensor): 包含图像的输入张量，形状为 (B, C, H, W)。
        bs (int): 批次大小，推导自 `im0` 的形状。
        mode (str): 当前处理模式，设置为 'image'。
        paths (List[str]): 图像路径列表或自动生成的文件名列表。

    方法：
        _single_check: 验证和格式化输入的张量。

    示例：
        >>> import torch
        >>> tensor = torch.rand(1, 3, 640, 640)
        >>> loader = LoadTensor(tensor)
        >>> paths, images, info = next(iter(loader))
        >>> print(f"处理了 {len(images)} 张图片")
    """

    def __init__(self, im0) -> None:
        """初始化 LoadTensor 对象，用于处理 torch.Tensor 图像数据。"""
        self.im0 = self._single_check(im0)
        self.bs = self.im0.shape[0]
        self.mode = "image"
        self.paths = [getattr(im, "filename", f"image{i}.jpg") for i, im in enumerate(im0)]

    @staticmethod
    def _single_check(im, stride=32):
        """验证和格式化单个图像张量，确保其形状和归一化正确。"""
        s = (
            f"警告 ⚠️ torch.Tensor 输入应为 BCHW 形状，即 shape(1, 3, 640, 640)，"
            f"且应能被 stride {stride} 整除。输入形状{tuple(im.shape)} 不兼容。"
        )
        if len(im.shape) != 4:
            if len(im.shape) != 3:
                raise ValueError(s)
            LOGGER.warning(s)
            im = im.unsqueeze(0)
        if im.shape[2] % stride or im.shape[3] % stride:
            raise ValueError(s)
        if im.max() > 1.0 + torch.finfo(im.dtype).eps:  # torch.float32 的 eps 是 1.2e-07
            LOGGER.warning(
                f"警告 ⚠️ torch.Tensor 输入应归一化到 0.0-1.0 范围，但最大值为 {im.max()}。"
                f"正在将输入除以 255."
            )
            im = im.float() / 255.0

        return im

    def __iter__(self):
        """返回一个迭代器对象，用于遍历张量图像数据。"""
        self.count = 0
        return self

    def __next__(self):
        """返回下一个批次的张量图像及其元数据供处理。"""
        if self.count == 1:
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, [""] * self.bs

    def __len__(self):
        """返回张量输入的批次大小。"""
        return self.bs


def autocast_list(source):
    """将多个源合并为一个包含 numpy 数组或 PIL 图像的列表，用于 Ultralytics 预测。"""
    files = []
    for im in source:
        if isinstance(im, (str, Path)):  # 文件名或 URI
            files.append(Image.open(requests.get(im, stream=True).raw if str(im).startswith("http") else im))
        elif isinstance(im, (Image.Image, np.ndarray)):  # PIL 或 np 图像
            files.append(im)
        else:
            raise TypeError(
                f"类型 {type(im).__name__} 不是支持的 Ultralytics 预测源类型。 \n"
                f"请参阅 https://docs.ultralytics.com/modes/predict 了解支持的源类型。"
            )

    return files


def get_best_youtube_url(url, method="pytube"):
    """
    获取给定 YouTube 视频的最佳质量 MP4 视频流的 URL。

    参数：
        url (str): YouTube 视频的 URL。
        method (str): 用于提取视频信息的方法。可选值为 "pytube"、"pafy" 和 "yt-dlp"。
            默认为 "pytube"。

    返回：
        (str | None): 最佳质量 MP4 视频流的 URL，如果未找到合适的流则返回 None。

    示例：
        >>> url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        >>> best_url = get_best_youtube_url(url)
        >>> print(best_url)
        https://rr4---sn-q4flrnek.googlevideo.com/videoplayback?expire=...

    注意：
        - 根据所选方法，需要额外的库：pytubefix、pafy 或 yt-dlp。
        - 函数会优先选择至少 1080p 分辨率的流（如果可用）。
        - 对于 "yt-dlp" 方法，它会寻找具有视频编解码器、没有音频、并且扩展名为 *.mp4 的格式。
    """
    if method == "pytube":
        # 从 pytube 切换到 pytubefix 以解决 https://github.com/pytube/pytube/issues/1954
        check_requirements("pytubefix>=6.5.2")
        from pytubefix import YouTube

        streams = YouTube(url).streams.filter(file_extension="mp4", only_video=True)
        streams = sorted(streams, key=lambda s: s.resolution, reverse=True)  # 按分辨率排序流
        for stream in streams:
            if stream.resolution and int(stream.resolution[:-1]) >= 1080:  # 检查分辨率是否至少为 1080p
                return stream.url

    elif method == "pafy":
        check_requirements(("pafy", "youtube_dl==2020.12.2"))
        import pafy  # noqa

        return pafy.new(url).getbestvideo(preftype="mp4").url

    elif method == "yt-dlp":
        check_requirements("yt-dlp")
        import yt_dlp

        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)  # 提取信息
        for f in reversed(info_dict.get("formats", [])):  # 因为最佳通常在最后，所以逆序处理
            # 查找具有视频编解码器、没有音频、扩展名为 *.mp4 且分辨率至少为 1920x1080 的格式
            good_size = (f.get("width") or 0) >= 1920 or (f.get("height") or 0) >= 1080
            if good_size and f["vcodec"] != "none" and f["acodec"] == "none" and f["ext"] == "mp4":
                return f.get("url")


# 定义常量
LOADERS = (LoadStreams, LoadPilAndNumpy, LoadImagesAndVideos, LoadScreenshots)

