# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np

from ..utils import LOGGER
from ..utils.ops import xywh2ltwh
from .basetrack import BaseTrack, TrackState
from .utils import matching
from .utils.kalman_filter import KalmanFilterXYAH


class STrack(BaseTrack):
    """
    使用卡尔曼滤波进行状态估计的单目标跟踪表示类。

    此类用于存储单个跟踪轨迹（tracklet）的所有相关信息，并基于卡尔曼滤波执行状态预测与更新。

    属性:
        shared_kalman (KalmanFilterXYAH): 所有 STrack 实例共享的卡尔曼滤波器。
        _tlwh (np.ndarray): 用于存储边界框的左上角坐标及宽高的私有变量。
        kalman_filter (KalmanFilterXYAH): 当前轨迹所使用的卡尔曼滤波器实例。
        mean (np.ndarray): 状态估计的均值向量。
        covariance (np.ndarray): 状态估计的协方差矩阵。
        is_activated (bool): 轨迹是否已激活的标志。
        score (float): 检测的置信度得分。
        tracklet_len (int): 当前轨迹的长度。
        cls (Any): 目标的类别标签。
        idx (int): 目标的编号或索引。
        frame_id (int): 当前帧的 ID。
        start_frame (int): 该轨迹首次被检测到的帧编号。

    方法:
        predict(): 使用卡尔曼滤波器预测下一帧的状态。
        multi_predict(stracks): 对多个轨迹同时执行预测。
        multi_gmc(stracks, H): 使用单应性矩阵更新多个轨迹的状态。
        activate(kalman_filter, frame_id): 激活一个新轨迹。
        re_activate(new_track, frame_id, new_id): 重新激活一个先前丢失的轨迹。
        update(new_track, frame_id): 更新一个已匹配的轨迹。
        convert_coords(tlwh): 将边界框转换为 (x, y, aspect, height) 格式。
        tlwh_to_xyah(tlwh): 将 tlwh 边界框格式转换为 xyah 格式。

    示例:
        初始化并激活一个新轨迹
        >>> track = STrack(xywh=[100, 200, 50, 80, 0], score=0.9, cls="person")
        >>> track.activate(kalman_filter=KalmanFilterXYAH(), frame_id=1)
    """

    shared_kalman = KalmanFilterXYAH()

    def __init__(self, xywh, score, cls):
        """
        初始化一个新的 STrack 实例。

        参数:
            xywh (List[float]): 边界框的中心坐标和尺寸，格式为 (x, y, w, h, [a], idx)，其中 (x, y) 为中心，
                (w, h) 为宽高，[a] 为可选的宽高比，idx 为编号。
            score (float): 检测的置信度得分。
            cls (Any): 检测到的目标类别。

        示例:
            >>> xywh = [100.0, 150.0, 50.0, 75.0, 1]
            >>> score = 0.9
            >>> cls = "person"
            >>> track = STrack(xywh, score, cls)
        """
        super().__init__()
        # xywh+idx 或 xywha+idx
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1]
        self.angle = xywh[4] if len(xywh) == 6 else None

    def predict(self):
        """使用卡尔曼滤波器预测目标的下一帧状态（均值和协方差）"""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """对多个 STrack 实例同时使用卡尔曼滤波器执行多目标预测。"""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """使用单应矩阵（H）对多个轨迹的位置和协方差进行更新。"""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """使用提供的卡尔曼滤波器激活一个新轨迹，并初始化其状态与协方差。"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

def re_activate(self, new_track, frame_id, new_id=False):
    """使用新的检测数据重新激活先前丢失的轨迹，并更新其状态和属性。"""
    self.mean, self.covariance = self.kalman_filter.update(
        self.mean, self.covariance, self.convert_coords(new_track.tlwh)
    )
    self.tracklet_len = 0
    self.state = TrackState.Tracked
    self.is_activated = True
    self.frame_id = frame_id
    if new_id:
        self.track_id = self.next_id()
    self.score = new_track.score
    self.cls = new_track.cls
    self.angle = new_track.angle
    self.idx = new_track.idx

def update(self, new_track, frame_id):
    """
    更新已匹配轨迹的状态。

    参数:
        new_track (STrack): 包含更新信息的新轨迹对象。
        frame_id (int): 当前帧的编号。

    示例:
        使用新的检测信息更新轨迹状态
        >>> track = STrack([100, 200, 50, 80, 0.9, 1])
        >>> new_track = STrack([105, 205, 55, 85, 0.95, 1])
        >>> track.update(new_track, 2)
    """
    self.frame_id = frame_id
    self.tracklet_len += 1

    new_tlwh = new_track.tlwh
    self.mean, self.covariance = self.kalman_filter.update(
        self.mean, self.covariance, self.convert_coords(new_tlwh)
    )
    self.state = TrackState.Tracked
    self.is_activated = True

    self.score = new_track.score
    self.cls = new_track.cls
    self.angle = new_track.angle
    self.idx = new_track.idx

def convert_coords(self, tlwh):
    """将边界框的左上角-宽高格式转换为中心点-x-y-长宽比-高度格式。"""
    return self.tlwh_to_xyah(tlwh)

@property
def tlwh(self):
    """从当前状态估计中返回边界框的左上角-宽度-高度格式。"""
    if self.mean is None:
        return self._tlwh.copy()
    ret = self.mean[:4].copy()
    ret[2] *= ret[3]
    ret[:2] -= ret[2:] / 2
    return ret

@property
def xyxy(self):
    """将边界框从 (左上角x, 左上角y, 宽度, 高度) 转换为 (最小x, 最小y, 最大x, 最大y) 格式。"""
    ret = self.tlwh.copy()
    ret[2:] += ret[:2]
    return ret

@staticmethod
def tlwh_to_xyah(tlwh):
    """将边界框从 tlwh 格式转换为中心点x-y-长宽比-高度 (xyah) 格式。"""
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    ret[2] /= ret[3]
    return ret

@property
def xywh(self):
    """返回当前边界框的位置，格式为 (中心x, 中心y, 宽度, 高度)。"""
    ret = np.asarray(self.tlwh).copy()
    ret[:2] += ret[2:] / 2
    return ret

@property
def xywha(self):
    """返回边界框的 (中心x, 中心y, 宽度, 高度, 角度) 格式，如果没有角度信息则警告并返回 xywh。"""
    if self.angle is None:
        LOGGER.warning("⚠️ 警告：未找到 `angle` 属性，返回 `xywh` 代替。")
        return self.xywh
    return np.concatenate([self.xywh, self.angle[None]])

@property
def result(self):
    """返回当前的跟踪结果，格式为合适的边界框格式。"""
    coords = self.xyxy if self.angle is None else self.xywha
    return coords.tolist() + [self.track_id, self.score, self.cls, self.idx]

def __repr__(self):
    """返回 STrack 对象的字符串表示，包含起始帧、结束帧和轨迹 ID。"""
    return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class BYTETracker:
    """
    BYTETracker：一个基于 YOLOv8 构建的目标检测与跟踪算法。

    负责在视频序列中初始化、更新和管理检测到的目标的跟踪轨迹。
    它在多帧之间维持已跟踪、丢失和移除的轨迹状态，利用 Kalman 滤波预测目标的新位置，并进行数据关联。

    属性:
        tracked_stracks (List[STrack]): 已成功激活的跟踪轨迹列表。
        lost_stracks (List[STrack]): 丢失的轨迹列表。
        removed_stracks (List[STrack]): 被移除的轨迹列表。
        frame_id (int): 当前帧的编号。
        args (Namespace): 命令行参数。
        max_time_lost (int): 一条轨迹被认为“丢失”前的最大帧数。
        kalman_filter (KalmanFilterXYAH): Kalman 滤波器对象。

    方法:
        update(results, img=None): 用新的检测结果更新目标跟踪器。
        get_kalmanfilter(): 返回一个用于边界框跟踪的 Kalman 滤波器对象。
        init_track(dets, scores, cls, img=None): 使用检测结果初始化目标跟踪。
        get_dists(tracks, detections): 计算轨迹与检测结果之间的距离。
        multi_predict(tracks): 预测轨迹的位置。
        reset_id(): 重置 STrack 的 ID 计数器。
        joint_stracks(tlista, tlistb): 合并两个轨迹列表。
        sub_stracks(tlista, tlistb): 从第一个列表中排除第二个列表中存在的轨迹。
        remove_duplicate_stracks(stracksa, stracksb): 基于 IoU 去除重复轨迹。

    示例:
        初始化 BYTETracker 并用检测结果进行更新：
        >>> tracker = BYTETracker(args, frame_rate=30)
        >>> results = yolo_model.detect(image)
        >>> tracked_objects = tracker.update(results)
    """

    def __init__(self, args, frame_rate=30):
        """
        初始化一个 BYTETracker 实例用于目标跟踪。

        参数:
            args (Namespace): 包含跟踪参数的命令行参数对象。
            frame_rate (int): 视频序列的帧率。

        示例:
            用命令行参数和帧率 30 初始化 BYTETracker：
            >>> args = Namespace(track_buffer=30)
            >>> tracker = BYTETracker(args, frame_rate=30)
        """
        self.tracked_stracks = []  # 当前被成功跟踪的轨迹列表（STrack 类型）
        self.lost_stracks = []  # 暂时丢失但保留的轨迹列表
        self.removed_stracks = []  # 被移除的轨迹列表

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)  # 允许目标丢失的最大时间（帧数）
        self.kalman_filter = self.get_kalmanfilter()  # 获取卡尔曼滤波器实例
        self.reset_id()


    def update(self, results, img=None):
        """使用新检测结果更新跟踪器，并返回当前帧中被跟踪的目标列表。"""
        self.frame_id += 1
        activated_stracks = []  # 当前帧被激活的轨迹
        refind_stracks = []  # 重新被找到的轨迹
        lost_stracks = []  # 在当前帧中被标记为丢失的轨迹
        removed_stracks = []  # 被移除的轨迹

        scores = results.conf
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        # 添加索引
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls

        # 根据高置信度阈值筛选检测框
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        # 得到中等置信度的检测框索引
        inds_second = inds_low & inds_high
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_keep = cls[remain_inds]
        cls_second = cls[inds_second]

        # 初始化高置信度检测的轨迹
        detections = self.init_track(dets, scores_keep, cls_keep, img)

        # 收集未确认（未激活）和已确认的轨迹
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # 第一步匹配：使用高置信度检测框与当前轨迹匹配
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)

        # 使用卡尔曼滤波器预测轨迹位置
        self.multi_predict(strack_pool)

        # 如果启用 GMC（全局运动补偿），则应用变换
        if hasattr(self, "gmc") and img is not None:
            warp = self.gmc.apply(img, dets)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        # 计算匹配距离并执行线性分配
        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 第二步匹配：使用中等置信度的检测框与未匹配轨迹进行匹配
        detections_second = self.init_track(dets_second, scores_second, cls_second, img)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        # 使用 IOU 进行匹配
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 未匹配上的轨迹，如果仍在跟踪中，则标记为丢失
        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # 处理未确认的轨迹，通常是只出现过一帧的轨迹
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # 第三步：初始化新的轨迹
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # 第四步：更新长期未匹配上的轨迹状态
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # 第五步：更新当前的状态列表
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)

        # 限制移除列表的长度不超过 1000
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]

        # 返回当前帧中处于激活状态的轨迹结果（float32）
        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def get_kalmanfilter(self):
        """返回一个用于边界框跟踪的 KalmanFilterXYAH 卡尔曼滤波器对象。"""
        return KalmanFilterXYAH()

    def init_track(self, dets, scores, cls, img=None):
        """使用提供的检测框、分数和类别标签，通过 STrack 算法初始化目标跟踪。"""
        return [STrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)] if len(dets) else []  # 检测框列表

    def get_dists(self, tracks, detections):
        """通过 IoU 计算跟踪轨迹与检测框之间的距离，并根据需要融合置信分数。"""
        dists = matching.iou_distance(tracks, detections)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks):
        """使用卡尔曼滤波器对多个轨迹预测下一个状态。"""
        STrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """重置 STrack 实例的 ID 计数器，以确保不同跟踪会话中轨迹 ID 的唯一性。"""
        STrack.reset_id()

    def reset(self):
        """重置跟踪器，清空所有已跟踪、丢失和移除的轨迹，并重新初始化卡尔曼滤波器。"""
        # type: list[STrack] 已跟踪目标列表
        self.tracked_stracks = []  
        # type: list[STrack] 丢失目标列表
        self.lost_stracks = []     
        # type: list[STrack] 被移除目标列表
        self.removed_stracks = []  
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    @staticmethod
    def joint_stracks(tlista, tlistb):
        """合并两个 STrack 对象列表，确保根据轨迹 ID 不重复。"""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        """从第一个列表中排除第二个列表中已存在的轨迹。"""
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """基于 IoU 距离从两个轨迹列表中移除重复的 STrack。"""
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
