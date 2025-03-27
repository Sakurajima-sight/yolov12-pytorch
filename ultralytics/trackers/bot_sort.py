# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import deque

import numpy as np

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH


class BOTrack(STrack):
    """
    YOLOv8 的 STrack 类的扩展版本，增加了目标跟踪功能。

    该类扩展了 STrack，添加了如特征平滑、卡尔曼预测、轨迹重激活等功能。

    属性:
        shared_kalman (KalmanFilterXYWH): 所有 BOTrack 实例共享的卡尔曼滤波器。
        smooth_feat (np.ndarray): 平滑后的特征向量。
        curr_feat (np.ndarray): 当前帧的特征向量。
        features (deque): 存储特征向量的队列，最大长度由 feat_history 定义。
        alpha (float): 用于特征指数平滑的系数。
        mean (np.ndarray): 卡尔曼滤波器的均值状态。
        covariance (np.ndarray): 卡尔曼滤波器的协方差矩阵。

    方法:
        update_features(feat): 更新特征向量，并使用指数滑动平均进行平滑。
        predict(): 使用卡尔曼滤波器预测均值和协方差。
        re_activate(new_track, frame_id, new_id): 使用新特征重新激活目标轨迹，可选更换 ID。
        update(new_track, frame_id): 使用新轨迹和帧 ID 更新当前对象。
        tlwh: 返回当前目标的 tlwh 格式坐标（左上角x, 左上角y, 宽, 高）。
        multi_predict(stracks): 使用共享卡尔曼滤波器对多个轨迹进行预测。
        convert_coords(tlwh): 将 tlwh 坐标格式转换为 xywh 格式。
        tlwh_to_xywh(tlwh): 将边界框从 tlwh 转换为 xywh 格式（中心x, 中心y, 宽, 高）。

    示例:
        创建一个 BOTrack 实例并更新其特征
        >>> bo_track = BOTrack(tlwh=[100, 50, 80, 40], score=0.9, cls=1, feat=np.random.rand(128))
        >>> bo_track.predict()
        >>> new_track = BOTrack(tlwh=[110, 60, 80, 40], score=0.85, cls=1, feat=np.random.rand(128))
        >>> bo_track.update(new_track, frame_id=2)
    """

    shared_kalman = KalmanFilterXYWH()

    def __init__(self, tlwh, score, cls, feat=None, feat_history=50):
        """
        初始化一个 BOTrack 对象，包含特征历史、平滑系数等时间相关参数。

        参数:
            tlwh (np.ndarray): 边界框坐标，格式为 (左上x, 左上y, 宽, 高)。
            score (float): 该检测的置信度分数。
            cls (int): 检测目标的类别 ID。
            feat (np.ndarray | None): 与该检测相关联的特征向量。
            feat_history (int): 特征历史记录队列的最大长度。

        示例:
            初始化一个 BOTrack 实例
            >>> tlwh = np.array([100, 50, 80, 120])
            >>> score = 0.9
            >>> cls = 1
            >>> feat = np.random.rand(128)
            >>> bo_track = BOTrack(tlwh, score, cls, feat)
        """
        super().__init__(tlwh, score, cls)

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        """更新当前帧的特征向量，并进行指数滑动平均平滑处理。"""
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        """使用卡尔曼滤波器预测目标的未来状态（均值和协方差）。"""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def re_activate(self, new_track, frame_id, new_id=False):
        """使用新轨迹更新当前轨迹并重激活目标，可选择是否更换 ID。"""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().re_activate(new_track, frame_id, new_id)

    def update(self, new_track, frame_id):
        """使用新轨迹信息和当前帧 ID 更新当前目标。"""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().update(new_track, frame_id)

    @property
    def tlwh(self):
        """返回当前目标的边界框位置，格式为 `(左上角x, 左上角y, 宽, 高)`。"""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks):
        """使用共享的卡尔曼滤波器对多个目标轨迹进行均值和协方差预测。"""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def convert_coords(self, tlwh):
        """将 tlwh 边界框坐标转换为 xywh 格式（中心点）。"""
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """将边界框从 tlwh（左上角）格式转换为 xywh（中心点）格式。"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret


class BOTSORT(BYTETracker):
    """
    BOTSORT 是 BYTETracker 类的扩展版本，适用于 YOLOv8，支持使用 ReID 和 GMC 算法进行目标跟踪。

    属性:
        proximity_thresh (float): 表示轨迹与检测框之间空间接近度（IoU）的阈值。
        appearance_thresh (float): 表示轨迹与检测框之间外观相似度（ReID特征）的阈值。
        encoder (Any): 处理 ReID 特征的对象，如果未启用 ReID，则为 None。
        gmc (GMC): GMC 算法实例，用于数据关联。
        args (Any): 包含跟踪参数的解析命令行参数。

    方法:
        get_kalmanfilter(): 返回一个 KalmanFilterXYWH 实例用于目标跟踪。
        init_track(dets, scores, cls, img): 使用检测结果、置信度分数和类别初始化轨迹。
        get_dists(tracks, detections): 使用 IoU（可选 ReID）计算轨迹与检测框之间的距离。
        multi_predict(tracks): 使用 YOLOv8 模型预测并跟踪多个目标。

    示例:
        初始化 BOTSORT 并处理检测结果
        >>> bot_sort = BOTSORT(args, frame_rate=30)
        >>> bot_sort.init_track(dets, scores, cls, img)
        >>> bot_sort.multi_predict(tracks)

    注意:
        该类专为与 YOLOv8 检测模型配合使用而设计，ReID 功能需通过 args 显式启用。
    """

    def __init__(self, args, frame_rate=30):
        """
        初始化 YOLOv8 对象跟踪器，包含 ReID 模块和 GMC 算法。

        参数:
            args (object): 包含跟踪参数的解析命令行参数。
            frame_rate (int): 当前处理视频的帧率。

        示例:
            使用命令行参数和指定帧率初始化 BOTSORT：
            >>> args = parse_args()
            >>> bot_sort = BOTSORT(args, frame_rate=30)
        """
        super().__init__(args, frame_rate)
        # ReID 模块
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        if args.with_reid:
            # 尚未支持带 ReID 的 BoT-SORT
            self.encoder = None
        self.gmc = GMC(method=args.gmc_method)

    def get_kalmanfilter(self):
        """返回一个 KalmanFilterXYWH 实例，用于在跟踪过程中预测和更新目标状态。"""
        return KalmanFilterXYWH()

    def init_track(self, dets, scores, cls, img=None):
        """使用检测框、置信度分数、类别标签和可选的 ReID 特征初始化目标轨迹。"""
        if len(dets) == 0:
            return []
        if self.args.with_reid and self.encoder is not None:
            features_keep = self.encoder.inference(img, dets)
            return [BOTrack(xyxy, s, c, f) for (xyxy, s, c, f) in zip(dets, scores, cls, features_keep)]  # 检测目标
        else:
            return [BOTrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)]  # 检测目标

    def get_dists(self, tracks, detections):
        """使用 IoU 和可选的 ReID 特征计算轨迹与检测目标之间的距离。"""
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > self.proximity_thresh

        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)

        if self.args.with_reid and self.encoder is not None:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)
        return dists

    def multi_predict(self, tracks):
        """使用共享的卡尔曼滤波器对多个目标轨迹进行状态预测。"""
        BOTrack.multi_predict(tracks)

    def reset(self):
        """重置 BOTSORT 跟踪器为初始状态，清除所有已跟踪目标和内部状态。"""
        super().reset()
        self.gmc.reset_params()
