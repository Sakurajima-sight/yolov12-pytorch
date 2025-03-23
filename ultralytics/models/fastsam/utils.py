# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license


def adjust_bboxes_to_image_border(boxes, image_shape, threshold=20):
    """
    如果边界框位于某个阈值内，则调整边界框使其贴合图像边界。

    参数：
        boxes (torch.Tensor): (n, 4)，边界框坐标。
        image_shape (tuple): (height, width)，图像的高度和宽度。
        threshold (int): 像素阈值。

    返回：
        adjusted_boxes (torch.Tensor): 调整后的边界框。
    """
    # 图像尺寸
    h, w = image_shape

    # 调整边界框
    boxes[boxes[:, 0] < threshold, 0] = 0  # x1
    boxes[boxes[:, 1] < threshold, 1] = 0  # y1
    boxes[boxes[:, 2] > w - threshold, 2] = w  # x2
    boxes[boxes[:, 3] > h - threshold, 3] = h  # y2
    return boxes
