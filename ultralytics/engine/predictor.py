# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
在图像、视频、目录、通配符、YouTube、网络摄像头、流等上运行预测。

使用方式 - 来源:
    $ yolo mode=predict model=yolov8n.pt source=0                               # 网络摄像头
                                                img.jpg                         # 图像
                                                vid.mp4                         # 视频
                                                screen                          # 截屏
                                                path/                           # 目录
                                                list.txt                        # 图像列表
                                                list.streams                    # 流列表
                                                'path/*.jpg'                    # 通配符
                                                'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP, TCP 流

使用方式 - 格式:
    $ yolo mode=predict model=yolov8n.pt                 # PyTorch
                              yolov8n.torchscript        # TorchScript
                              yolov8n.onnx               # ONNX Runtime 或 OpenCV DNN，dnn=True
                              yolov8n_openvino_model     # OpenVINO
                              yolov8n.engine             # TensorRT
                              yolov8n.mlpackage          # CoreML (仅限 macOS)
                              yolov8n_saved_model        # TensorFlow SavedModel
                              yolov8n.pb                 # TensorFlow GraphDef
                              yolov8n.tflite             # TensorFlow Lite
                              yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov8n_paddle_model       # PaddlePaddle
                              yolov8n.mnn                # MNN
                              yolov8n_ncnn_model         # NCNN
"""

import platform
import re
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox, classify_transforms
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.checks import check_imgsz, check_imshow
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import select_device, smart_inference_mode

STREAM_WARNING = """
警告 ⚠️ 如果没有传递`stream=True`，推理结果将会累积到内存中，可能会导致大源或长时间运行的流和视频出现内存不足错误。详情请参考：https://docs.ultralytics.com/modes/predict/

示例：
    results = model(source=..., stream=True)  # 结果对象的生成器
    for r in results:
        boxes = r.boxes  # 框对象，用于边界框输出
        masks = r.masks  # 掩码对象，用于分割掩码输出
        probs = r.probs  # 类别概率，用于分类输出
"""



class BasePredictor:
    """
    BasePredictor。

    一个用于创建预测器的基类。

    属性:
        args (SimpleNamespace): 预测器的配置。
        save_dir (Path): 用于保存结果的目录。
        done_warmup (bool): 预测器是否已完成初始化。
        model (nn.Module): 用于预测的模型。
        data (dict): 数据配置。
        device (torch.device): 用于预测的设备。
        dataset (Dataset): 用于预测的数据集。
        vid_writer (dict): 用于保存视频输出的字典，键为保存路径，值为视频写入器。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        初始化 BasePredictor 类。

        参数:
            cfg (str, 可选): 配置文件的路径。默认值为 DEFAULT_CFG。
            overrides (dict, 可选): 配置的覆盖。默认值为 None。
        """
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        if self.args.conf is None:
            self.args.conf = 0.25  # 默认 conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # 如果设置完成则可用
        self.model = None
        self.data = self.args.data  # 数据字典
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_writer = {}  # {保存路径: 视频写入器, ...}
        self.plotted_img = None
        self.source_type = None
        self.seen = 0
        self.windows = []
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self._lock = threading.Lock()  # 用于自动线程安全推理
        callbacks.add_integration_callbacks(self)

    def preprocess(self, im):
        """
        在推理前准备输入图像。

        参数:
            im (torch.Tensor | List(np.ndarray)): BCHW格式的张量，或[(HWC) x B]格式的列表。

        返回:
            (torch.Tensor): 预处理后的图像。
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR转RGB，BHWC转BCHW，(n, 3, h, w)
            im = np.ascontiguousarray(im)  # 连续数组
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8转fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 转到 0.0 - 1.0
        return im

    def inference(self, im, *args, **kwargs):
        """使用指定的模型和参数对给定图像进行推理。"""
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and (not self.source_type.tensor)
            else False
        )
        return self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)

    def pre_transform(self, im):
        """
        在推理前对输入图像进行预处理。

        参数:
            im (List(np.ndarray)): (N, 3, h, w) 对于张量，[(h, w, 3) x N] 对于列表。

        返回:
            (list): 预处理后的图像列表。
        """
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(
            self.imgsz,
            auto=same_shapes and (self.model.pt or (getattr(self.model, "dynamic", False) and not self.model.imx)),
            stride=self.model.stride,
        )
        return [letterbox(image=x) for x in im]

    def postprocess(self, preds, img, orig_imgs):
        """对图像的预测结果进行后处理并返回。"""
        return preds

    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        """对图像或流进行推理。"""
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))  # 将结果合并为一个

    def predict_cli(self, source=None, model=None):
        """
        用于命令行界面（CLI）预测的方法。

        该函数用于通过CLI运行预测。它设置源和模型，然后以流式方式处理输入。
        该方法确保在预测过程中不积累输出，避免内存问题。

        注意:
            不要修改此函数或移除生成器。生成器确保不会在内存中积累输出，
            这对于长时间运行的预测至关重要。
        """
        gen = self.stream_inference(source, model)
        for _ in gen:  # sourcery skip: remove-empty-nested-block, noqa
            pass

    def setup_source(self, source):
        """设置源和推理模式。"""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # 检查图像大小
        self.transforms = (
            getattr(
                self.model.model,
                "transforms",
                classify_transforms(self.imgsz[0], crop_fraction=self.args.crop_fraction),
            )
            if self.args.task == "classify"
            else None
        )
        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer,
        )
        self.source_type = self.dataset.source_type
        if not getattr(self, "stream", True) and (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # 图像数量过多
            or any(getattr(self.dataset, "video_flag", [False]))
        ):  # 视频
            LOGGER.warning(STREAM_WARNING)
        self.vid_writer = {}

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """在摄像头画面上进行实时推理并将结果保存到文件中。"""
        if self.args.verbose:
            LOGGER.info("")

        # 设置模型
        if not self.model:
            self.setup_model(model)

        with self._lock:  # 用于线程安全推理
            # 每次预测调用时设置源
            self.setup_source(source if source is not None else self.args.source)

            # 检查 save_dir/ 标签文件是否存在
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # 预热模型
            if not self.done_warmup:
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch

                # 预处理
                with profilers[0]:
                    im = self.preprocess(im0s)

                # 推理
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # 生成嵌入张量
                        continue

                # 后处理
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")

                # 可视化、保存、写入结果
                n = len(im0s)
                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess": profilers[0].dt * 1e3 / n,
                        "inference": profilers[1].dt * 1e3 / n,
                        "postprocess": profilers[2].dt * 1e3 / n,
                    }
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s[i] += self.write_results(i, Path(paths[i]), im, s)

                # 打印批次结果
                if self.args.verbose:
                    LOGGER.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results

        # 释放资源
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        # 打印最终结果
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # 每张图像的速度
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), 3, *im.shape[2:])}" % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # 标签数量
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")

    def setup_model(self, model, verbose=True):
        """初始化 YOLO 模型并设置为评估模式。"""
        self.model = AutoBackend(
            weights=model or self.args.model,
            device=select_device(self.args.device, verbose=verbose),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            batch=self.args.batch,
            fuse=True,
            verbose=verbose,
        )

        self.device = self.model.device  # 更新设备
        self.args.half = self.model.fp16  # 更新 fp16
        self.model.eval()

    def write_results(self, i, p, im, s):
        """将推理结果写入文件或目录。"""
        string = ""  # 打印的字符串
        if len(im.shape) == 3:
            im = im[None]  # 扩展为批次维度
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # 如果帧号未确定则为0

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "{:g}x{:g} ".format(*im.shape[2:])
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()  # 用于其他位置
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"

        # 将预测结果添加到图像上
        if self.args.save or self.args.show:
            self.plotted_img = result.plot(
                line_width=self.args.line_width,
                boxes=self.args.show_boxes,
                conf=self.args.show_conf,
                labels=self.args.show_labels,
                im_gpu=None if self.args.retina_masks else im[i],
            )

        # 保存结果
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))
        if self.args.save:
            self.save_predicted_images(str(self.save_dir / p.name), frame)

        return string

    def save_predicted_images(self, save_path="", frame=0):
        """保存视频预测结果为mp4文件到指定路径。"""
        im = self.plotted_img

        # 保存视频和流
        if self.dataset.mode in {"stream", "video"}:
            fps = self.dataset.fps if self.dataset.mode == "video" else 30
            frames_path = f"{save_path.split('.', 1)[0]}_frames/"
            if save_path not in self.vid_writer:  # 新的视频
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=fps,  # 必须是整数，浮点数在MP4编码器中会报错
                    frameSize=(im.shape[1], im.shape[0]),  # (宽度, 高度)
                )

            # 保存视频
            self.vid_writer[save_path].write(im)
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}{frame}.jpg", im)

        # 保存图像
        else:
            cv2.imwrite(str(Path(save_path).with_suffix(".jpg")), im)  # 保存为JPG格式，最佳兼容性

    def show(self, p=""):
        """使用OpenCV的imshow函数在窗口中显示图像。"""
        im = self.plotted_img
        if platform.system() == "Linux" and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 允许窗口调整大小（Linux）
            cv2.resizeWindow(p, im.shape[1], im.shape[0])  # (宽度, 高度)
        cv2.imshow(p, im)
        cv2.waitKey(300 if self.dataset.mode == "image" else 1)  # 1 毫秒

    def run_callbacks(self, event: str):
        """运行为特定事件注册的所有回调函数。"""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func):
        """添加回调函数。"""
        self.callbacks[event].append(func)
