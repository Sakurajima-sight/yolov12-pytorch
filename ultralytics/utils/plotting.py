# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pil_version

from ultralytics.utils import IS_COLAB, IS_KAGGLE, LOGGER, TryExcept, ops, plt_settings, threaded
from ultralytics.utils.checks import check_font, check_version, is_ascii
from ultralytics.utils.files import increment_path


class Colors:
    """
    Ultralytics 颜色调色板 https://docs.ultralytics.com/reference/utils/plotting/#ultralytics.utils.plotting.Colors。

    该类提供了与 Ultralytics 颜色调色板相关的方法，包括将十六进制颜色代码转换为 RGB 值。

    属性:
        palette (list of tuple): RGB 颜色值的列表。
        n (int): 调色板中的颜色数量。
        pose_palette (np.ndarray): 一个具有 np.uint8 数据类型的特定颜色调色板数组。

    ## Ultralytics 颜色调色板

    | 索引 | 颜色                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #042aff;"></i> | `#042aff` | (4, 42, 255)      |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #0bdbeb;"></i> | `#0bdbeb` | (11, 219, 235)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #f3f3f3;"></i> | `#f3f3f3` | (243, 243, 243)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #00dfb7;"></i> | `#00dfb7` | (0, 223, 183)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #111f68;"></i> | `#111f68` | (17, 31, 104)     |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #ff6fdd;"></i> | `#ff6fdd` | (255, 111, 221)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff444f;"></i> | `#ff444f` | (255, 68, 79)     |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #cced00;"></i> | `#cced00` | (204, 237, 0)     |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #00f344;"></i> | `#00f344` | (0, 243, 68)      |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #bd00ff;"></i> | `#bd00ff` | (189, 0, 255)     |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #00b4ff;"></i> | `#00b4ff` | (0, 180, 255)     |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #dd00ba;"></i> | `#dd00ba` | (221, 0, 186)     |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #00ffff;"></i> | `#00ffff` | (0, 255, 255)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #26c000;"></i> | `#26c000` | (38, 192, 0)      |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #01ffb3;"></i> | `#01ffb3` | (1, 255, 179)     |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #7d24ff;"></i> | `#7d24ff` | (125, 36, 255)    |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #7b0068;"></i> | `#7b0068` | (123, 0, 104)     |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #ff1b6c;"></i> | `#ff1b6c` | (255, 27, 108)    |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #fc6d2f;"></i> | `#fc6d2f` | (252, 109, 47)    |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #a2ff0b;"></i> | `#a2ff0b` | (162, 255, 11)    |

    ## 姿势颜色调色板

    | 索引 | 颜色                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #ff8000;"></i> | `#ff8000` | (255, 128, 0)     |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #ff9933;"></i> | `#ff9933` | (255, 153, 51)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #ffb266;"></i> | `#ffb266` | (255, 178, 102)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #e6e600;"></i> | `#e6e600` | (230, 230, 0)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #ff99ff;"></i> | `#ff99ff` | (255, 153, 255)   |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #99ccff;"></i> | `#99ccff` | (153, 204, 255)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff66ff;"></i> | `#ff66ff` | (255, 102, 255)   |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #ff33ff;"></i> | `#ff33ff` | (255, 51, 255)    |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #66b2ff;"></i> | `#66b2ff` | (102, 178, 255)   |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #3399ff;"></i> | `#3399ff` | (51, 153, 255)    |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #ff9999;"></i> | `#ff9999` | (255, 153, 153)   |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #ff6666;"></i> | `#ff6666` | (255, 102, 102)   |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #ff3333;"></i> | `#ff3333` | (255, 51, 51)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #99ff99;"></i> | `#99ff99` | (153, 255, 153)   |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #66ff66;"></i> | `#66ff66` | (102, 255, 102)   |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #33ff33;"></i> | `#33ff33` | (51, 255, 51)     |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #00ff00;"></i> | `#00ff00` | (0, 255, 0)       |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #0000ff;"></i> | `#0000ff` | (0, 0, 255)       |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #ff0000;"></i> | `#ff0000` | (255, 0, 0)       |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #ffffff;"></i> | `#ffffff` | (255, 255, 255)   |

    !!! note "Ultralytics 品牌颜色"

        查看 Ultralytics 品牌颜色请访问 [https://www.ultralytics.com/brand](https://www.ultralytics.com/brand)。请在所有市场营销材料中使用官方的 Ultralytics 颜色。
    """

    def __init__(self):
        """初始化颜色值，作为 hex = matplotlib.colors.TABLEAU_COLORS.values()。"""
        hexs = (
            "042AFF",
            "0BDBEB",
            "F3F3F3",
            "00DFB7",
            "111F68",
            "FF6FDD",
            "FF444F",
            "CCED00",
            "00F344",
            "BD00FF",
            "00B4FF",
            "DD00BA",
            "00FFFF",
            "26C000",
            "01FFB3",
            "7D24FF",
            "7B0068",
            "FF1B6C",
            "FC6D2F",
            "A2FF0B",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """将十六进制颜色代码转换为 RGB 值。"""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """将十六进制颜色代码转换为 RGB 值（即默认的 PIL 顺序）。"""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # 创建实例，供 'from utils.plots import colors' 使用


class Annotator:
    """
    用于训练/验证拼接图像和JPG以及预测注释的 Ultralytics Annotator。

    属性：
        im (Image.Image 或 numpy 数组): 要进行注释的图像。
        pil (bool): 是否使用 PIL 或 cv2 来绘制注释。
        font (ImageFont.truetype 或 ImageFont.load_default): 用于文本注释的字体。
        lw (float): 绘制线条的宽度。
        skeleton (List[List[int]]): 关键点的骨架结构。
        limb_color (List[int]): 肢体的颜色调色板。
        kpt_color (List[int]): 关键点的颜色调色板。
    """

    def __init__(self, im, line_width=None, font_size=None, font="Arial.ttf", pil=False, example="abc"):
        """初始化 Annotator 类，传入图像、线宽以及关键点和肢体的颜色调色板。"""
        non_ascii = not is_ascii(example)  # 非拉丁标签，例如亚洲、阿拉伯、俄语
        input_is_pil = isinstance(im, Image.Image)
        self.pil = pil or non_ascii or input_is_pil
        self.lw = line_width or max(round(sum(im.size if input_is_pil else im.shape) / 2 * 0.003), 2)
        if self.pil:  # 使用 PIL
            self.im = im if input_is_pil else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            try:
                font = check_font("Arial.Unicode.ttf" if non_ascii else font)
                size = font_size or max(round(sum(self.im.size) / 2 * 0.035), 12)
                self.font = ImageFont.truetype(str(font), size)
            except Exception:
                self.font = ImageFont.load_default()
            # 过时的修复：w, h = getsize(string) -> _, _, w, h = getbox(string)
            if check_version(pil_version, "9.2.0"):
                self.font.getsize = lambda x: self.font.getbbox(x)[2:4]  # 文本宽度，高度
        else:  # 使用 cv2
            assert im.data.contiguous, "图像不是连续的。请对输入图像使用 np.ascontiguousarray(im)。"
            self.im = im if im.flags.writeable else im.copy()
            self.tf = max(self.lw - 1, 1)  # 字体厚度
            self.sf = self.lw / 3  # 字体比例
        # 姿态
        self.skeleton = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]

        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        self.dark_colors = {
            (235, 219, 11),
            (243, 243, 243),
            (183, 223, 0),
            (221, 111, 255),
            (0, 237, 204),
            (68, 243, 0),
            (255, 255, 0),
            (179, 255, 1),
            (11, 255, 162),
        }
        self.light_colors = {
            (255, 42, 4),
            (79, 68, 255),
            (255, 0, 189),
            (255, 180, 0),
            (186, 0, 221),
            (0, 192, 38),
            (255, 36, 125),
            (104, 0, 123),
            (108, 27, 255),
            (47, 109, 252),
            (104, 31, 17),
        }

    def get_txt_color(self, color=(128, 128, 128), txt_color=(255, 255, 255)):
        """
        根据背景颜色分配文本颜色。

        参数：
            color (tuple, 可选): 文本矩形的背景颜色（B, G, R）。
            txt_color (tuple, 可选): 文本的颜色（R, G, B）。

        返回：
            txt_color (tuple): 文本标签的颜色
        """
        if color in self.dark_colors:
            return 104, 31, 17
        elif color in self.light_colors:
            return 255, 255, 255
        else:
            return txt_color

    def circle_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), margin=2):
        """
        在给定边界框内绘制带有背景圆圈的标签。

        参数：
            box (tuple): 边界框的坐标 (x1, y1, x2, y2)。
            label (str): 要显示的文本标签。
            color (tuple, 可选): 矩形背景的颜色（B, G, R）。
            txt_color (tuple, 可选): 文本的颜色（R, G, B）。
            margin (int, 可选): 文本与矩形边框之间的间距。
        """
        # 如果标签超过 3 个字符，跳过其他字符，以适应圆圈大小
        if len(label) > 3:
            print(
                f"标签长度为 {len(label)}，将只考虑前 3 个字符用于圆圈注释！"
            )
            label = label[:3]

        # 计算框的中心
        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        # 获取文本的大小
        text_size = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.15, self.tf)[0]
        # 计算所需的半径以适应带有间距的文本
        required_radius = int(((text_size[0] ** 2 + text_size[1] ** 2) ** 0.5) / 2) + margin
        # 绘制具有所需半径的圆圈
        cv2.circle(self.im, (x_center, y_center), required_radius, color, -1)
        # 计算文本位置
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2
        # 绘制文本
        cv2.putText(
            self.im,
            str(label),
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.sf - 0.15,
            self.get_txt_color(color, txt_color),
            self.tf,
            lineType=cv2.LINE_AA,
        )

    def text_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), margin=5):
        """
        在给定的边界框内绘制带背景矩形的标签。

        参数：
            box (tuple): 边界框坐标 (x1, y1, x2, y2)。
            label (str): 要显示的文本标签。
            color (tuple, 可选): 矩形的背景颜色 (B, G, R)，默认为 (128, 128, 128)。
            txt_color (tuple, 可选): 文本的颜色 (R, G, B)，默认为 (255, 255, 255)。
            margin (int, 可选): 文本与矩形边框之间的间距，默认为 5。
        """
        # 计算边界框的中心
        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        # 获取文本的尺寸
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.1, self.tf)[0]
        # 计算文本的左上角坐标（以便居中）
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2
        # 计算背景矩形的坐标
        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[0] + margin
        rect_y2 = text_y + margin
        # 绘制背景矩形
        cv2.rectangle(self.im, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)
        # 在矩形上绘制文本
        cv2.putText(
            self.im,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.sf - 0.1,
            self.get_txt_color(color, txt_color),
            self.tf,
            lineType=cv2.LINE_AA,
        )

    def box_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), rotated=False):
        """
        在图像上绘制边界框及标签。

        参数：
            box (tuple): 边界框坐标 (x1, y1, x2, y2)。
            label (str): 要显示的文本标签。
            color (tuple, 可选): 矩形的背景颜色 (B, G, R)，默认为 (128, 128, 128)。
            txt_color (tuple, 可选): 文本的颜色 (R, G, B)，默认为 (255, 255, 255)。
            rotated (bool, 可选): 用于检查任务是否为 OBB（有向边界框）的变量。
        """
        txt_color = self.get_txt_color(color, txt_color)
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        if self.pil or not is_ascii(label):
            if rotated:
                p1 = box[0]
                self.draw.polygon([tuple(b) for b in box], width=self.lw, outline=color)  # PIL 需要元组格式的边界框
            else:
                p1 = (box[0], box[1])
                self.draw.rectangle(box, width=self.lw, outline=color)  # 绘制矩形
            if label:
                w, h = self.font.getsize(label)  # 获取文本宽度和高度
                outside = p1[1] >= h  # 检查标签是否在框外
                if p1[0] > self.im.size[0] - w:  # 检查标签是否超出图像右侧
                    p1 = self.im.size[0] - w, p1[1]
                self.draw.rectangle(
                    (p1[0], p1[1] - h if outside else p1[1], p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1),
                    fill=color,
                )
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # PIL>8.0
                self.draw.text((p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font)
        else:  # 使用 cv2
            if rotated:
                p1 = [int(b) for b in box[0]]
                cv2.polylines(self.im, [np.asarray(box, dtype=int)], True, color, self.lw)  # cv2 需要 nparray 格式的边界框
            else:
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]  # 获取文本宽度和高度
                h += 3  # 为文本增加像素边距
                outside = p1[1] >= h  # 检查标签是否在框外
                if p1[0] > self.im.shape[1] - w:  # 检查标签是否超出图像右侧
                    p1 = self.im.shape[1] - w, p1[1]
                p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # 填充矩形
                cv2.putText(
                    self.im,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h - 1),
                    0,
                    self.sf,
                    txt_color,
                    thickness=self.tf,
                    lineType=cv2.LINE_AA,
                )

    def masks(self, masks, colors, im_gpu, alpha=0.5, retina_masks=False):
        """
        在图像上绘制掩模。

        参数：
            masks (tensor): 预测的掩模，形状为 [n, h, w]，在 cuda 上。
            colors (List[List[Int]]): 预测掩模的颜色，形状为 [[r, g, b] * n]。
            im_gpu (tensor): 图像在 cuda 上，形状为 [3, h, w]，范围为 [0, 1]。
            alpha (float): 掩模透明度：0.0 完全透明，1.0 完全不透明。
            retina_masks (bool): 是否使用高分辨率掩模，默认为 False。
        """
        if self.pil:
            # 首先转换为 numpy 数组
            self.im = np.asarray(self.im).copy()
        if len(masks) == 0:
            self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
        if im_gpu.device != masks.device:
            im_gpu = im_gpu.to(masks.device)
        colors = torch.tensor(colors, device=masks.device, dtype=torch.float32) / 255.0  # 形状为(n,3)
        colors = colors[:, None, None]  # 形状为(n,1,1,3)
        masks = masks.unsqueeze(3)  # 形状为(n,h,w,1)
        masks_color = masks * (colors * alpha)  # 形状为(n,h,w,3)

        inv_alpha_masks = (1 - masks * alpha).cumprod(0)  # 形状为(n,h,w,1)
        mcs = masks_color.max(dim=0).values  # 形状为(n,h,w,3)

        im_gpu = im_gpu.flip(dims=[0])  # 翻转通道
        im_gpu = im_gpu.permute(1, 2, 0).contiguous()  # 形状为(h,w,3)
        im_gpu = im_gpu * inv_alpha_masks[-1] + mcs
        im_mask = im_gpu * 255
        im_mask_np = im_mask.byte().cpu().numpy()
        self.im[:] = im_mask_np if retina_masks else ops.scale_image(im_mask_np, self.im.shape)
        if self.pil:
            # 将图像转换回 PIL 格式并更新绘制
            self.fromarray(self.im) 

    def kpts(self, kpts, shape=(640, 640), radius=None, kpt_line=True, conf_thres=0.25, kpt_color=None):
        """
        在图像上绘制关键点。

        参数：
            kpts (torch.Tensor): 关键点，形状为 [17, 3] (x, y, confidence)。
            shape (tuple, 可选): 图像的形状 (h, w)。默认为 (640, 640)。
            radius (int, 可选): 关键点的半径。默认为 5。
            kpt_line (bool, 可选): 是否在关键点之间绘制连线。默认为 True。
            conf_thres (float, 可选): 置信度阈值。默认为 0.25。
            kpt_color (tuple, 可选): 关键点颜色 (B, G, R)。默认为 None。

        注意：
            - `kpt_line=True` 目前仅支持人体姿态绘制。
            - 原地修改 self.im。
            - 如果 self.pil 为 True，则将图像从 numpy 数组转换回 PIL 图像。
        """
        radius = radius if radius is not None else self.lw
        if self.pil:
            # 首先转换为 numpy 数组
            self.im = np.asarray(self.im).copy()
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim in {2, 3}
        kpt_line &= is_pose  # `kpt_line=True` 目前仅支持人体姿态绘制
        for i, k in enumerate(kpts):
            color_k = kpt_color or (self.kpt_color[i].tolist() if is_pose else colors(i))
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < conf_thres:
                        continue
                cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

        if kpt_line:
            ndim = kpts.shape[-1]
            for i, sk in enumerate(self.skeleton):
                pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                if ndim == 3:
                    conf1 = kpts[(sk[0] - 1), 2]
                    conf2 = kpts[(sk[1] - 1), 2]
                    if conf1 < conf_thres or conf2 < conf_thres:
                        continue
                if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                    continue
                if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue
                cv2.line(
                    self.im,
                    pos1,
                    pos2,
                    kpt_color or self.limb_color[i].tolist(),
                    thickness=int(np.ceil(self.lw / 2)),
                    lineType=cv2.LINE_AA,
                )
        if self.pil:
            # 将 im 转换回 PIL 并更新绘制
            self.fromarray(self.im)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        """在图像上添加矩形（仅限 PIL）。"""
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor="top", box_style=False):
        """使用 PIL 或 cv2 向图像添加文本。"""
        if anchor == "bottom":  # 从字体底部开始 y 坐标
            w, h = self.font.getsize(text)  # 文本的宽度和高度
            xy[1] += 1 - h
        if self.pil:
            if box_style:
                w, h = self.font.getsize(text)
                self.draw.rectangle((xy[0], xy[1], xy[0] + w + 1, xy[1] + h + 1), fill=txt_color)
                # 使用 `txt_color` 作为背景色，并以白色绘制前景
                txt_color = (255, 255, 255)
            if "\n" in text:
                lines = text.split("\n")
                _, h = self.font.getsize(text)
                for line in lines:
                    self.draw.text(xy, line, fill=txt_color, font=self.font)
                    xy[1] += h
            else:
                self.draw.text(xy, text, fill=txt_color, font=self.font)
        else:
            if box_style:
                w, h = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]  # 文本宽度和高度
                h += 3  # 增加像素以填充文本
                outside = xy[1] >= h  # 标签是否适合框外
                p2 = xy[0] + w, xy[1] - h if outside else xy[1] + h
                cv2.rectangle(self.im, xy, p2, txt_color, -1, cv2.LINE_AA)  # 填充矩形
                # 使用 `txt_color` 作为背景色，并以白色绘制前景
                txt_color = (255, 255, 255)
            cv2.putText(self.im, text, xy, 0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA)

    def fromarray(self, im):
        """从 numpy 数组更新 self.im。"""
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        """返回带注释的图像作为数组。"""
        return np.asarray(self.im)

    def show(self, title=None):
        """显示带注释的图像。"""
        im = Image.fromarray(np.asarray(self.im)[..., ::-1])  # 将 numpy 数组转换为 PIL 图像，并将 RGB 转换为 BGR
        if IS_COLAB or IS_KAGGLE:  # 不能使用 IS_JUPYTER，因为它将在所有 ipython 环境中运行
            try:
                display(im)  # noqa - display() 仅在 ipython 环境中可用
            except ImportError as e:
                LOGGER.warning(f"无法在 Jupyter 笔记本中显示图像: {e}")
        else:
            im.show(title=title)

    def save(self, filename="image.jpg"):
        """将带注释的图像保存到 'filename'。"""
        cv2.imwrite(filename, np.asarray(self.im))

    @staticmethod
    def get_bbox_dimension(bbox=None):
        """
        计算边界框的面积。

        参数：
            bbox (tuple): 边界框坐标，格式为 (x_min, y_min, x_max, y_max)。

        返回：
            width (float): 边界框的宽度。
            height (float): 边界框的高度。
            area (float): 边界框的面积。
        """
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        return width, height, width * height

    def draw_region(self, reg_pts=None, color=(0, 255, 0), thickness=5):
        """
        绘制区域线。

        参数:
            reg_pts (list): 区域点（对于线是 2 个点，对于区域是 4 个点）
            color (tuple): 区域颜色值
            thickness (int): 区域厚度值
        """
        cv2.polylines(self.im, [np.array(reg_pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)

        # 在角点绘制小圆圈
        for point in reg_pts:
            cv2.circle(self.im, (point[0], point[1]), thickness * 2, color, -1)  # -1 表示填充圆圈

    def draw_centroid_and_tracks(self, track, color=(255, 0, 255), track_thickness=2):
        """
        绘制质心点和轨迹。

        参数:
            track (list): 用于显示轨迹的物体跟踪点
            color (tuple): 轨迹线的颜色
            track_thickness (int): 轨迹线的厚度
        """
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(self.im, [points], isClosed=False, color=color, thickness=track_thickness)
        cv2.circle(self.im, (int(track[-1][0]), int(track[-1][1])), track_thickness * 2, color, -1)

    def queue_counts_display(self, label, points=None, region_color=(255, 255, 255), txt_color=(0, 0, 0)):
        """
        在图像上显示队列计数，文本居中显示，支持自定义字体大小和颜色。

        参数:
            label (str): 队列计数标签。
            points (tuple): 用于计算中心点的区域点，用于显示文本。
            region_color (tuple): 队列区域的颜色（RGB）。
            txt_color (tuple): 文本显示的颜色（RGB）。
        """
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        center_x = sum(x_values) // len(points)
        center_y = sum(y_values) // len(points)

        text_size = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
        text_width = text_size[0]
        text_height = text_size[1]

        rect_width = text_width + 20
        rect_height = text_height + 20
        rect_top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
        rect_bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)
        cv2.rectangle(self.im, rect_top_left, rect_bottom_right, region_color, -1)

        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2

        # 绘制文本
        cv2.putText(
            self.im,
            label,
            (text_x, text_y),
            0,
            fontScale=self.sf,
            color=txt_color,
            thickness=self.tf,
            lineType=cv2.LINE_AA,
        )

    def display_objects_labels(self, im0, text, txt_color, bg_color, x_center, y_center, margin):
        """
        在停车管理应用中显示边界框标签。

        参数:
            im0 (ndarray): 推理图像。
            text (str): 物体/类别名称。
            txt_color (tuple): 文本前景颜色。
            bg_color (tuple): 文本背景颜色。
            x_center (float): 边界框的 x 中心点位置。
            y_center (float): 边界框的 y 中心点位置。
            margin (int): 文本与矩形之间的间隙，用于更好的显示。
        """
        text_size = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2

        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[0] + margin
        rect_y2 = text_y + margin
        cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
        cv2.putText(im0, text, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)

    def display_analytics(self, im0, text, txt_color, bg_color, margin):
        """
        显示停车场的整体统计信息。

        参数:
            im0 (ndarray): 推理图像。
            text (dict): 标签字典。
            txt_color (tuple): 文本前景颜色。
            bg_color (tuple): 文本背景颜色。
            margin (int): 文本与矩形之间的间隙，用于更好的显示。
        """
        horizontal_gap = int(im0.shape[1] * 0.02)
        vertical_gap = int(im0.shape[0] * 0.01)
        text_y_offset = 0
        for label, value in text.items():
            txt = f"{label}: {value}"
            text_size = cv2.getTextSize(txt, 0, self.sf, self.tf)[0]
            if text_size[0] < 5 or text_size[1] < 5:
                text_size = (5, 5)
            text_x = im0.shape[1] - text_size[0] - margin * 2 - horizontal_gap
            text_y = text_y_offset + text_size[1] + margin * 2 + vertical_gap
            rect_x1 = text_x - margin * 2
            rect_y1 = text_y - text_size[1] - margin * 2
            rect_x2 = text_x + text_size[0] + margin * 2
            rect_y2 = text_y + margin * 2
            cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
            cv2.putText(im0, txt, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)
            text_y_offset = rect_y2

    @staticmethod
    def estimate_pose_angle(a, b, c):
        """
        计算物体的姿态角度。

        参数：
            a (float) : 姿态点 a 的值
            b (float): 姿态点 b 的值
            c (float): 姿态点 c 的值

        返回：
            angle (degree): 三个点之间的角度（以度为单位）
        """
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def draw_specific_points(self, keypoints, indices=None, radius=2, conf_thres=0.25):
        """
        绘制用于健身计数的特定关键点。

        参数：
            keypoints (list): 要绘制的关键点数据。
            indices (list, 可选): 要绘制的关键点索引。默认为 [2, 5, 7]。
            radius (int, 可选): 关键点半径。默认为 2。
            conf_thres (float, 可选): 关键点的置信度阈值。默认为 0.25。

        返回：
            (numpy.ndarray): 绘制了关键点的图像。

        注意：
            关键点格式： [x, y] 或 [x, y, confidence]。
            在原地修改 self.im。
        """
        indices = indices or [2, 5, 7]
        points = [(int(k[0]), int(k[1])) for i, k in enumerate(keypoints) if i in indices and k[2] >= conf_thres]

        # 在连续的点之间绘制线条
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(self.im, start, end, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # 绘制关键点的圆形
        for pt in points:
            cv2.circle(self.im, pt, radius, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        return self.im

    def plot_workout_information(self, display_text, position, color=(104, 31, 17), txt_color=(255, 255, 255)):
        """
        在图像上绘制带背景的文本。

        参数：
            display_text (str): 要显示的文本。
            position (tuple): 文本在图像上的位置坐标 (x, y)。
            color (tuple, 可选): 文本背景色。
            txt_color (tuple, 可选): 文本前景色。
        """
        (text_width, text_height), _ = cv2.getTextSize(display_text, 0, self.sf, self.tf)

        # 绘制背景矩形
        cv2.rectangle(
            self.im,
            (position[0], position[1] - text_height - 5),
            (position[0] + text_width + 10, position[1] - text_height - 5 + text_height + 10 + self.tf),
            color,
            -1,
        )
        # 绘制文本
        cv2.putText(self.im, display_text, position, 0, self.sf, txt_color, self.tf)

        return text_height

    def plot_angle_and_count_and_stage(
        self, angle_text, count_text, stage_text, center_kpt, color=(104, 31, 17), txt_color=(255, 255, 255)
    ):
        """
        绘制姿态角度、计数值和阶段信息。

        参数：
            angle_text (str): 用于健身监控的角度值。
            count_text (str): 用于健身监控的计数值。
            stage_text (str): 用于健身监控的阶段信息。
            center_kpt (list): 用于健身监控的质心姿态索引。
            color (tuple, 可选): 文本背景色。
            txt_color (tuple, 可选): 文本前景色。
        """
        # 格式化文本
        angle_text, count_text, stage_text = f" {angle_text:.2f}", f"Steps : {count_text}", f" {stage_text}"

        # 绘制角度、计数和阶段文本
        angle_height = self.plot_workout_information(
            angle_text, (int(center_kpt[0]), int(center_kpt[1])), color, txt_color
        )
        count_height = self.plot_workout_information(
            count_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + 20), color, txt_color
        )
        self.plot_workout_information(
            stage_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + count_height + 40), color, txt_color
        )

    def seg_bbox(self, mask, mask_color=(255, 0, 255), label=None, txt_color=(255, 255, 255)):
        """
        在边界框形状中绘制分割对象。

        参数：
            mask (np.ndarray): 形状为 (N, 2) 的二维数组，包含分割对象的轮廓点。
            mask_color (tuple): 轮廓和标签背景的 RGB 颜色。
            label (str, 可选): 对象的文本标签。如果为 None，则不绘制标签。
            txt_color (tuple): 标签文本的 RGB 颜色。
        """
        if mask.size == 0:  # 没有需要绘制的掩码
            return

        cv2.polylines(self.im, [np.int32([mask])], isClosed=True, color=mask_color, thickness=2)
        text_size, _ = cv2.getTextSize(label, 0, self.sf, self.tf)

        if label:
            cv2.rectangle(
                self.im,
                (int(mask[0][0]) - text_size[0] // 2 - 10, int(mask[0][1]) - text_size[1] - 10),
                (int(mask[0][0]) + text_size[0] // 2 + 10, int(mask[0][1] + 10)),
                mask_color,
                -1,
            )
            cv2.putText(
                self.im, label, (int(mask[0][0]) - text_size[0] // 2, int(mask[0][1])), 0, self.sf, txt_color, self.tf
            )

    def sweep_annotator(self, line_x=0, line_y=0, label=None, color=(221, 0, 186), txt_color=(255, 255, 255)):
        """
        绘制扫线注释线和可选的标签。

        参数:
            line_x (int): 扫线的 x 坐标。
            line_y (int): 扫线的 y 坐标限制。
            label (str, 可选): 在扫线中心绘制的文本标签。如果为 None，则不绘制标签。
            color (tuple): 线条和标签背景的 RGB 颜色。
            txt_color (tuple): 标签文本的 RGB 颜色。
        """
        # 绘制扫线
        cv2.line(self.im, (line_x, 0), (line_x, line_y), color, self.tf * 2)

        # 如果提供了标签，则绘制标签
        if label:
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf, self.tf)
            cv2.rectangle(
                self.im,
                (line_x - text_width // 2 - 10, line_y // 2 - text_height // 2 - 10),
                (line_x + text_width // 2 + 10, line_y // 2 + text_height // 2 + 10),
                color,
                -1,
            )
            cv2.putText(
                self.im,
                label,
                (line_x - text_width // 2, line_y // 2 + text_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.sf,
                txt_color,
                self.tf,
            )

    def plot_distance_and_line(
        self, pixels_distance, centroids, line_color=(104, 31, 17), centroid_color=(255, 0, 255)
    ):
        """
        在图像上绘制距离和连线。

        参数:
            pixels_distance (float): 两个边界框质心之间的像素距离。
            centroids (list): 边界框质心数据。
            line_color (tuple, 可选): 距离线的颜色。
            centroid_color (tuple, 可选): 边界框质心的颜色。
        """
        # 获取文本大小
        text = f"Pixels Distance: {pixels_distance:.2f}"
        (text_width_m, text_height_m), _ = cv2.getTextSize(text, 0, self.sf, self.tf)

        # 定义带有 10 像素间隔的矩形并绘制
        cv2.rectangle(self.im, (15, 25), (15 + text_width_m + 20, 25 + text_height_m + 20), line_color, -1)

        # 计算文本位置并绘制文本
        text_position = (25, 25 + text_height_m + 10)
        cv2.putText(
            self.im,
            text,
            text_position,
            0,
            self.sf,
            (255, 255, 255),
            self.tf,
            cv2.LINE_AA,
        )

        # 绘制质心之间的线
        cv2.line(self.im, centroids[0], centroids[1], line_color, 3)
        cv2.circle(self.im, centroids[0], 6, centroid_color, -1)
        cv2.circle(self.im, centroids[1], 6, centroid_color, -1)

    def visioneye(self, box, center_point, color=(235, 219, 11), pin_color=(255, 0, 255)):
        """
        用于标定人眼视角映射和绘图。

        参数:
            box (list): 边界框坐标
            center_point (tuple): 视角中心点
            color (tuple): 物体质心和线条的颜色值
            pin_color (tuple): 视角点的颜色值
        """
        center_bbox = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        cv2.circle(self.im, center_point, self.tf * 2, pin_color, -1)
        cv2.circle(self.im, center_bbox, self.tf * 2, color, -1)
        cv2.line(self.im, center_point, center_bbox, color, self.tf)


@TryExcept()  # 已知问题 https://github.com/ultralytics/yolov5/issues/5395
@plt_settings()
def plot_labels(boxes, cls, names=(), save_dir=Path(""), on_plot=None):
    """绘制训练标签，包括类别直方图和边框统计信息。"""
    import pandas  # 为了更快地导入 'ultralytics'
    import seaborn  # 为了更快地导入 'ultralytics'

    # 过滤 matplotlib>=3.7.2 警告和 Seaborn use_inf 与 is_categorical FutureWarnings
    warnings.filterwarnings("ignore", category=UserWarning, message="The figure layout has changed to tight")
    warnings.filterwarnings("ignore", category=FutureWarning)

    # 绘制数据集标签
    LOGGER.info(f"正在将标签绘制到 {save_dir / 'labels.jpg'}... ")
    nc = int(cls.max() + 1)  # 类别数
    boxes = boxes[:1000000]  # 限制为 100 万个边界框
    x = pandas.DataFrame(boxes, columns=["x", "y", "width", "height"])

    # Seaborn correlogram
    seaborn.pairplot(x, corner=True, diag_kind="auto", kind="hist", diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / "labels_correlogram.jpg", dpi=200)
    plt.close()

    # Matplotlib 标签
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(cls, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    for i in range(nc):
        y[2].patches[i].set_color([x / 255 for x in colors(i)])
    ax[0].set_ylabel("实例数")
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel("类别")
    seaborn.histplot(x, x="x", y="y", ax=ax[2], bins=50, pmax=0.9)
    seaborn.histplot(x, x="width", y="height", ax=ax[3], bins=50, pmax=0.9)

    # 绘制矩形
    boxes[:, 0:2] = 0.5  # 中心
    boxes = ops.xywh2xyxy(boxes) * 1000
    img = Image.fromarray(np.ones((1000, 1000, 3), dtype=np.uint8) * 255)
    for cls, box in zip(cls[:500], boxes[:500]):
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # 绘制
    ax[1].imshow(img)
    ax[1].axis("off")

    for a in [0, 1, 2, 3]:
        for s in ["top", "right", "left", "bottom"]:
            ax[a].spines[s].set_visible(False)

    fname = save_dir / "labels.jpg"
    plt.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)


def save_one_box(xyxy, im, file=Path("im.jpg"), gain=1.02, pad=10, square=False, BGR=False, save=True):
    """
    将图像裁剪并保存为 {file}，裁剪框的大小按 {gain} 和 {pad} 像素调整。保存和/或返回裁剪的图像。

    该函数接受一个边界框和图像，然后根据边界框保存裁剪部分的图像。
    可选地，裁剪可以变为正方形，并且该函数允许调整边界框的增益和填充。

    参数：
        xyxy (torch.Tensor 或 list): 表示边界框的张量或列表，格式为 xyxy。
        im (numpy.ndarray): 输入图像。
        file (Path, 可选): 裁剪图像保存的路径，默认为 'im.jpg'。
        gain (float, 可选): 一个乘数因子，用于增大边界框的大小，默认为 1.02。
        pad (int, 可选): 要添加到边界框宽度和高度的像素数，默认为 10。
        square (bool, 可选): 如果为 True，边界框将变为正方形，默认为 False。
        BGR (bool, 可选): 如果为 True，图像将以 BGR 格式保存，否则以 RGB 格式保存，默认为 False。
        save (bool, 可选): 如果为 True，裁剪后的图像将保存到磁盘，默认为 True。

    返回：
        (numpy.ndarray): 裁剪后的图像。

    示例：
        ```python
        from ultralytics.utils.plotting import save_one_box

        xyxy = [50, 50, 150, 150]
        im = cv2.imread("image.jpg")
        cropped_im = save_one_box(xyxy, im, file="cropped.jpg", square=True)
        ```
    """
    if not isinstance(xyxy, torch.Tensor):  # 可能是列表
        xyxy = torch.stack(xyxy)
    b = ops.xyxy2xywh(xyxy.view(-1, 4))  # 边界框
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # 尝试将矩形转为正方形
    b[:, 2:] = b[:, 2:] * gain + pad  # 边界框宽高 * gain + pad
    xyxy = ops.xywh2xyxy(b).long()
    xyxy = ops.clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # 创建目录
        f = str(increment_path(file).with_suffix(".jpg"))
        # cv2.imwrite(f, crop)  # 保存为 BGR, https://github.com/ultralytics/yolov5/issues/7007 色度子采样问题
        Image.fromarray(crop[..., ::-1]).save(f, quality=95, subsampling=0)  # 保存为 RGB
    return crop


@threaded
def plot_images(
    images: Union[torch.Tensor, np.ndarray],
    batch_idx: Union[torch.Tensor, np.ndarray],
    cls: Union[torch.Tensor, np.ndarray],
    bboxes: Union[torch.Tensor, np.ndarray] = np.zeros(0, dtype=np.float32),
    confs: Optional[Union[torch.Tensor, np.ndarray]] = None,
    masks: Union[torch.Tensor, np.ndarray] = np.zeros(0, dtype=np.uint8),
    kpts: Union[torch.Tensor, np.ndarray] = np.zeros((0, 51), dtype=np.float32),
    paths: Optional[List[str]] = None,
    fname: str = "images.jpg",
    names: Optional[Dict[int, str]] = None,
    on_plot: Optional[Callable] = None,
    max_size: int = 1920,
    max_subplots: int = 16,
    save: bool = True,
    conf_thres: float = 0.25,
) -> Optional[np.ndarray]:
    """
    绘制包含标签、边界框、掩模和关键点的图像网格。

    参数：
        images: 要绘制的图像批次，形状为 (batch_size, channels, height, width)。
        batch_idx: 每个检测的批次索引，形状为 (num_detections,)。
        cls: 每个检测的类别标签，形状为 (num_detections,)。
        bboxes: 每个检测的边界框，形状为 (num_detections, 4) 或 (num_detections, 5)（旋转框）。
        confs: 每个检测的置信度分数，形状为 (num_detections,)。
        masks: 实例分割掩模，形状为 (num_detections, height, width) 或 (1, height, width)。
        kpts: 每个检测的关键点，形状为 (num_detections, 51)。
        paths: 每张图像的文件路径列表。
        fname: 输出图像网格的文件名。
        names: 类别索引到类别名称的字典。
        on_plot: 可选的回调函数，在保存图像后调用。
        max_size: 输出图像网格的最大大小。
        max_subplots: 图像网格中最多的子图数量。
        save: 是否将绘制的图像网格保存到文件。
        conf_thres: 显示检测的置信度阈值。

    返回：
        np.ndarray: 如果 save 为 False，返回绘制的图像网格作为 numpy 数组，否则返回 None。

    注意：
        该函数支持 tensor 和 numpy 数组输入。它会自动将 tensor 输入转换为 numpy 数组进行处理。
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(cls, torch.Tensor):
        cls = cls.cpu().numpy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy().astype(int)
    if isinstance(kpts, torch.Tensor):
        kpts = kpts.cpu().numpy()
    if isinstance(batch_idx, torch.Tensor):
        batch_idx = batch_idx.cpu().numpy()

    bs, _, h, w = images.shape  # 批次大小，_，高度，宽度
    bs = min(bs, max_subplots)  # 限制绘制的图像数量
    ns = np.ceil(bs**0.5)  # 子图的数量（以正方形的方式）
    if np.max(images[0]) <= 1:
        images *= 255  # 反归一化（可选）

    # 构建图像
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # 初始化
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # 块的原点
        mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0)

    # 调整大小（可选）
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # 注释
    fs = int((h + w) * ns * 0.01)  # 字体大小
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # 块的原点
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # 边框
        if paths:
            annotator.text((x + 5, y + 5), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # 文件名
        if len(cls) > 0:
            idx = batch_idx == i
            classes = cls[idx].astype("int")
            labels = confs is None

            if len(bboxes):
                boxes = bboxes[idx]
                conf = confs[idx] if confs is not None else None  # 检查置信度是否存在（标签与预测）
                if len(boxes):
                    if boxes[:, :4].max() <= 1.1:  # 如果是归一化的，容忍度为 0.1
                        boxes[..., [0, 2]] *= w  # 缩放到像素
                        boxes[..., [1, 3]] *= h
                    elif scale < 1:  # 如果图像被缩放，则绝对坐标需要缩放
                        boxes[..., :4] *= scale
                boxes[..., 0] += x
                boxes[..., 1] += y
                is_obb = boxes.shape[-1] == 5  # xywhr
                boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)
                for j, box in enumerate(boxes.astype(np.int64).tolist()):
                    c = classes[j]
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    if labels or conf[j] > conf_thres:
                        label = f"{c}" if labels else f"{c} {conf[j]:.1f}"
                        annotator.box_label(box, label, color=color, rotated=is_obb)

            elif len(classes):
                for c in classes:
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    annotator.text((x, y), f"{c}", txt_color=color, box_style=True)

            # 绘制关键点
            if len(kpts):
                kpts_ = kpts[idx].copy()
                if len(kpts_):
                    if kpts_[..., 0].max() <= 1.01 or kpts_[..., 1].max() <= 1.01:  # 如果是归一化的，容忍度为 .01
                        kpts_[..., 0] *= w  # 缩放到像素
                        kpts_[..., 1] *= h
                    elif scale < 1:  # 如果图像被缩放，则绝对坐标需要缩放
                        kpts_ *= scale
                kpts_[..., 0] += x
                kpts_[..., 1] += y
                for j in range(len(kpts_)):
                    if labels or conf[j] > conf_thres:
                        annotator.kpts(kpts_[j], conf_thres=conf_thres)

            # 绘制掩模
            if len(masks):
                if idx.shape[0] == masks.shape[0]:  # overlap_masks=False
                    image_masks = masks[idx]
                else:  # overlap_masks=True
                    image_masks = masks[[i]]  # (1, 640, 640)
                    nl = idx.sum()
                    index = np.arange(nl).reshape((nl, 1, 1)) + 1
                    image_masks = np.repeat(image_masks, nl, axis=0)
                    image_masks = np.where(image_masks == index, 1.0, 0.0)

                im = np.asarray(annotator.im).copy()
                for j in range(len(image_masks)):
                    if labels or conf[j] > conf_thres:
                        color = colors(classes[j])
                        mh, mw = image_masks[j].shape
                        if mh != h or mw != w:
                            mask = image_masks[j].astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            mask = mask.astype(bool)
                        else:
                            mask = image_masks[j].astype(bool)
                        try:
                            im[y : y + h, x : x + w, :][mask] = (
                                im[y : y + h, x : x + w, :][mask] * 0.4 + np.array(color) * 0.6
                            )
                        except Exception:
                            pass
                annotator.fromarray(im)
    if not save:
        return np.asarray(annotator.im)
    annotator.im.save(fname)  # 保存
    if on_plot:
        on_plot(fname)


@plt_settings()
def plot_results(file="path/to/results.csv", dir="", segment=False, pose=False, classify=False, on_plot=None):
    """
    从结果 CSV 文件绘制训练结果。该函数支持多种类型的数据，包括分割、姿态估计和分类。图表将保存在 CSV 所在的目录下，文件名为 'results.png'。

    参数：
        file (str, 可选): 包含训练结果的 CSV 文件路径。默认为 'path/to/results.csv'。
        dir (str, 可选): 如果未提供 'file'，则为 CSV 文件所在的目录。默认为 ''。
        segment (bool, 可选): 标志，指示数据是否用于分割。默认为 False。
        pose (bool, 可选): 标志，指示数据是否用于姿态估计。默认为 False。
        classify (bool, 可选): 标志，指示数据是否用于分类。默认为 False。
        on_plot (callable, 可选): 绘图后执行的回调函数。接收文件名作为参数。默认为 None。

    示例：
        ```python
        from ultralytics.utils.plotting import plot_results

        plot_results("path/to/results.csv", segment=True)
        ```
    """
    import pandas as pd  # 为加速 'import ultralytics' 使用作用域
    from scipy.ndimage import gaussian_filter1d

    save_dir = Path(file).parent if file else Path(dir)
    if classify:
        fig, ax = plt.subplots(2, 2, figsize=(6, 6), tight_layout=True)
        index = [2, 5, 3, 4]
    elif segment:
        fig, ax = plt.subplots(2, 8, figsize=(18, 6), tight_layout=True)
        index = [2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16, 17, 8, 9, 12, 13]
    elif pose:
        fig, ax = plt.subplots(2, 9, figsize=(21, 6), tight_layout=True)
        index = [2, 3, 4, 5, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 9, 10, 13, 14]
    else:
        fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
        index = [2, 3, 4, 5, 6, 9, 10, 11, 7, 8]
    ax = ax.ravel()
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"未找到 {save_dir.resolve()} 中的 results.csv 文件，无法绘图。"
    for f in files:
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate(index):
                y = data.values[:, j].astype("float")
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=8)  # 绘制实际结果
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":", label="平滑", linewidth=2)  # 平滑曲线
                ax[i].set_title(s[j], fontsize=12)
        except Exception as e:
            LOGGER.warning(f"警告: 绘制 {f} 时出错: {e}")
    ax[1].legend()
    fname = save_dir / "results.png"
    fig.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)


def plt_color_scatter(v, f, bins=20, cmap="viridis", alpha=0.8, edgecolors="none"):
    """
    绘制一个基于二维直方图着色的散点图。

    参数：
        v (array-like): 用于 x 轴的值。
        f (array-like): 用于 y 轴的值。
        bins (int, 可选): 直方图的箱子数量。默认为 20。
        cmap (str, 可选): 散点图的色图。默认为 'viridis'。
        alpha (float, 可选): 散点图的透明度。默认为 0.8。
        edgecolors (str, 可选): 散点图的边缘颜色。默认为 'none'。

    示例：
        >>> v = np.random.rand(100)
        >>> f = np.random.rand(100)
        >>> plt_color_scatter(v, f)
    """
    # 计算二维直方图及其对应的颜色
    hist, xedges, yedges = np.histogram2d(v, f, bins=bins)
    colors = [
        hist[
            min(np.digitize(v[i], xedges, right=True) - 1, hist.shape[0] - 1),
            min(np.digitize(f[i], yedges, right=True) - 1, hist.shape[1] - 1),
        ]
        for i in range(len(v))
    ]

    # 绘制散点图
    plt.scatter(v, f, c=colors, cmap=cmap, alpha=alpha, edgecolors=edgecolors)


def plot_tune_results(csv_file="tune_results.csv"):
    """
    绘制保存在 'tune_results.csv' 文件中的演化结果。该函数为 CSV 中的每个键生成一个散点图，并根据适应度得分进行颜色编码。最佳表现的配置将在图表中突出显示。

    参数：
        csv_file (str, 可选): 包含调优结果的 CSV 文件路径。默认为 'tune_results.csv'。

    示例：
        >>> plot_tune_results("path/to/tune_results.csv")
    """
    import pandas as pd  # 为加速 'import ultralytics' 使用作用域
    from scipy.ndimage import gaussian_filter1d

    def _save_one_file(file):
        """将一个 matplotlib 图保存到 'file'。"""
        plt.savefig(file, dpi=200)
        plt.close()
        LOGGER.info(f"已保存 {file}")

    # 每个超参数的散点图
    csv_file = Path(csv_file)
    data = pd.read_csv(csv_file)
    num_metrics_columns = 1
    keys = [x.strip() for x in data.columns][num_metrics_columns:]
    x = data.values
    fitness = x[:, 0]  # 适应度
    j = np.argmax(fitness)  # 最大适应度的索引
    n = math.ceil(len(keys) ** 0.5)  # 图表中的列和行数
    plt.figure(figsize=(10, 10), tight_layout=True)
    for i, k in enumerate(keys):
        v = x[:, i + num_metrics_columns]
        mu = v[j]  # 最好的单一结果
        plt.subplot(n, n, i + 1)
        plt_color_scatter(v, fitness, cmap="viridis", alpha=0.8, edgecolors="none")
        plt.plot(mu, fitness.max(), "k+", markersize=15)
        plt.title(f"{k} = {mu:.3g}", fontdict={"size": 9})  # 限制为 40 个字符
        plt.tick_params(axis="both", labelsize=8)  # 设置轴标签大小为 8
        if i % n != 0:
            plt.yticks([])
    _save_one_file(csv_file.with_name("tune_scatter_plots.png"))

    # 适应度与迭代次数的关系图
    x = range(1, len(fitness) + 1)
    plt.figure(figsize=(10, 6), tight_layout=True)
    plt.plot(x, fitness, marker="o", linestyle="none", label="fitness")
    plt.plot(x, gaussian_filter1d(fitness, sigma=3), ":", label="平滑", linewidth=2)  # 平滑线
    plt.title("适应度 vs 迭代次数")
    plt.xlabel("迭代次数")
    plt.ylabel("适应度")
    plt.grid(True)
    plt.legend()
    _save_one_file(csv_file.with_name("tune_fitness.png"))


def output_to_target(output, max_det=300):
    """将模型输出转换为目标格式 [batch_id, class_id, x, y, w, h, conf] 用于绘制。"""
    targets = []
    for i, o in enumerate(output):
        box, conf, cls = o[:max_det, :6].cpu().split((4, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, ops.xyxy2xywh(box), conf), 1))
    targets = torch.cat(targets, 0).numpy()
    return targets[:, 0], targets[:, 1], targets[:, 2:-1], targets[:, -1]


def output_to_rotated_target(output, max_det=300):
    """将模型输出转换为目标格式 [batch_id, class_id, x, y, w, h, conf] 用于绘制。"""
    targets = []
    for i, o in enumerate(output):
        box, conf, cls, angle = o[:max_det].cpu().split((4, 1, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, box, angle, conf), 1))
    targets = torch.cat(targets, 0).numpy()
    return targets[:, 0], targets[:, 1], targets[:, 2:-1], targets[:, -1]


def feature_visualization(x, module_type, stage, n=32, save_dir=Path("runs/detect/exp")):
    """
    可视化给定模型模块在推理过程中的特征图。

    参数:
        x (torch.Tensor): 需要可视化的特征。
        module_type (str): 模块类型。
        stage (int): 模块在模型中的阶段。
        n (int, 可选): 最大可视化的特征图数量。默认为32。
        save_dir (Path, 可选): 结果保存目录。默认为 Path('runs/detect/exp')。
    """
    for m in {"Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder"}:  # 所有模型头
        if m in module_type:
            return
    if isinstance(x, torch.Tensor):
        _, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # 文件名

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # 选择batch索引0，按通道分块
            n = min(n, channels)  # 可视化的图像数量
            _, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8行 x n/8列
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # 使用灰度色图
                ax[i].axis("off")

            LOGGER.info(f"保存 {f}... ({n}/{channels})")
            plt.savefig(f, dpi=300, bbox_inches="tight")
            plt.close()
            np.save(str(f.with_suffix(".npy")), x[0].cpu().numpy())  # 保存为npy文件
