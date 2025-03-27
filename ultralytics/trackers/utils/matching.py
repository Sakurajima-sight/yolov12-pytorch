# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np
import scipy
from scipy.spatial.distance import cdist

from ultralytics.utils.metrics import batch_probiou, bbox_ioa

try:
    import lap  # 用于线性分配

    assert lap.__version__  # 验证导入的是库不是目录
except (ImportError, AssertionError, AttributeError):
    from ultralytics.utils.checks import check_requirements

    check_requirements("lap>=0.5.12")  # https://github.com/gatagat/lap
    import lap


def linear_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = True) -> tuple:
    """
    使用 scipy 或 lap.lapjv 方法进行线性分配。

    参数:
        cost_matrix (np.ndarray): 包含代价的矩阵，形状为 (N, M)。
        thresh (float): 判断匹配是否有效的阈值。
        use_lap (bool): 是否使用 lap.lapjv 进行分配。如果为 False，则使用 scipy.optimize.linear_sum_assignment。

    返回:
        matched_indices (np.ndarray): 匹配的索引数组，形状为 (K, 2)，K 为匹配数量。
        unmatched_a (np.ndarray): 第一个集合中未匹配的索引，形状为 (L,)。
        unmatched_b (np.ndarray): 第二个集合中未匹配的索引，形状为 (M,)。

    示例:
        >>> cost_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> thresh = 5.0
        >>> matched_indices, unmatched_a, unmatched_b = linear_assignment(cost_matrix, thresh, use_lap=True)
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    if use_lap:
        # 使用 lap.lapjv 进行分配
        # https://github.com/gatagat/lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:
        # 使用 scipy.optimize.linear_sum_assignment
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        x, y = scipy.optimize.linear_sum_assignment(cost_matrix)  # 行 x，列 y
        matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])
        if len(matches) == 0:
            unmatched_a = list(np.arange(cost_matrix.shape[0]))
            unmatched_b = list(np.arange(cost_matrix.shape[1]))
        else:
            unmatched_a = list(set(np.arange(cost_matrix.shape[0])) - set(matches[:, 0]))
            unmatched_b = list(set(np.arange(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def iou_distance(atracks: list, btracks: list) -> np.ndarray:
    """
    基于 IoU（交并比）计算两个轨迹集之间的距离代价。

    参数:
        atracks (list[STrack] | list[np.ndarray]): 第一个轨迹集或边界框列表。
        btracks (list[STrack] | list[np.ndarray]): 第二个轨迹集或边界框列表。

    返回:
        (np.ndarray): 基于 IoU 计算得到的代价矩阵。

    示例:
        >>> atracks = [np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])]
        >>> btracks = [np.array([5, 5, 15, 15]), np.array([25, 25, 35, 35])]
        >>> cost_matrix = iou_distance(atracks, btracks)
    """
    if atracks and isinstance(atracks[0], np.ndarray) or btracks and isinstance(btracks[0], np.ndarray):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.xywha if track.angle is not None else track.xyxy for track in atracks]
        btlbrs = [track.xywha if track.angle is not None else track.xyxy for track in btracks]

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if len(atlbrs) and len(btlbrs):
        if len(atlbrs[0]) == 5 and len(btlbrs[0]) == 5:
            # 处理旋转框的 IoU 计算
            ious = batch_probiou(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
            ).numpy()
        else:
            # 普通边界框的 IoU 计算
            ious = bbox_ioa(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
                iou=True,
            )
    return 1 - ious  # 返回代价矩阵


def embedding_distance(tracks: list, detections: list, metric: str = "cosine") -> np.ndarray:
    """
    基于嵌入特征（embedding）计算轨迹和检测之间的距离。

    参数:
        tracks (list[STrack]): 包含嵌入特征的轨迹列表。
        detections (list[BaseTrack]): 包含嵌入特征的检测目标列表。
        metric (str): 距离计算使用的度量方式。支持 'cosine'、'euclidean' 等。

    返回:
        (np.ndarray): 基于嵌入特征计算得到的代价矩阵，形状为 (N, M)，N 为轨迹数，M 为检测数。

    示例:
        >>> cost_matrix = embedding_distance(tracks, detections, metric="cosine")
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # 特征已归一化
    return cost_matrix


def fuse_score(cost_matrix: np.ndarray, detections: list) -> np.ndarray:
    """
    将代价矩阵与检测得分进行融合，生成一个新的相似度矩阵。

    参数:
        cost_matrix (np.ndarray): 分配时的原始代价矩阵，形状为 (N, M)。
        detections (list[BaseTrack]): 检测对象列表，每个对象包含 score 属性。

    返回:
        (np.ndarray): 融合后的相似度矩阵，形状为 (N, M)。

    示例:
        >>> cost_matrix = np.random.rand(5, 10)  # 5 条轨迹和 10 个检测
        >>> detections = [BaseTrack(score=np.random.rand()) for _ in range(10)]
        >>> fused_matrix = fuse_score(cost_matrix, detections)
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim  # 返回融合后的代价矩阵
