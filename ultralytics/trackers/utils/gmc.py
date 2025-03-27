# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import copy

import cv2
import numpy as np

from ultralytics.utils import LOGGER


class GMC:
    """
    广义运动补偿（GMC）类，用于视频帧中的目标跟踪和检测。

    此类提供多种跟踪算法（包括 ORB、SIFT、ECC 和稀疏光流）的方法，用于基于帧之间的变化进行目标跟踪与检测。
    同时支持帧降采样以提高处理效率。

    属性:
        method (str): 所使用的跟踪方法，可选项包括 'orb', 'sift', 'ecc', 'sparseOptFlow', 'none'。
        downscale (int): 降采样因子，用于加快处理速度。
        prevFrame (np.ndarray): 存储前一帧图像。
        prevKeyPoints (List): 存储前一帧的关键点。
        prevDescriptors (np.ndarray): 存储前一帧的特征描述符。
        initializedFirstFrame (bool): 标志位，指示是否已经处理了第一帧。

    方法:
        __init__: 初始化 GMC 对象，指定跟踪方法和降采样因子。
        apply: 对输入帧应用所选方法，可选地使用提供的检测框。
        apply_ecc: 对输入帧应用 ECC 算法。
        apply_features: 使用 ORB 或 SIFT 等特征点方法处理帧。
        apply_sparseoptflow: 使用稀疏光流法处理帧。
        reset_params: 重置 GMC 对象的内部状态参数。

    示例:
        创建一个 GMC 对象并应用于一帧图像
        >>> gmc = GMC(method="sparseOptFlow", downscale=2)
        >>> frame = np.array([[1, 2, 3], [4, 5, 6]])
        >>> processed_frame = gmc.apply(frame)
        >>> print(processed_frame)
        array([[1, 2, 3],
               [4, 5, 6]])
    """

    def __init__(self, method: str = "sparseOptFlow", downscale: int = 2) -> None:
        """
        初始化 GMC（广义运动补偿）对象，指定跟踪方法和图像降采样因子。

        参数:
            method (str): 所选跟踪方法，支持 'orb', 'sift', 'ecc', 'sparseOptFlow', 'none'。
            downscale (int): 图像降采样因子，用于加快处理速度。

        示例:
            初始化一个使用稀疏光流方法，降采样因子为2的 GMC 对象
            >>> gmc = GMC(method="sparseOptFlow", downscale=2)
        """
        super().__init__()

        self.method = method
        self.downscale = max(1, downscale)

        if self.method == "orb":
            self.detector = cv2.FastFeatureDetector_create(20)
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        elif self.method == "sift":
            self.detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.extractor = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        elif self.method == "ecc":
            number_of_iterations = 5000
            termination_eps = 1e-6
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        elif self.method == "sparseOptFlow":
            self.feature_params = dict(
                maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3, useHarrisDetector=False, k=0.04
            )

        elif self.method in {"none", "None", None}:
            self.method = None
        else:
            raise ValueError(f"错误：未知的 GMC 方法: {method}")

        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False

    def apply(self, raw_frame: np.array, detections: list = None) -> np.array:
        """
        对输入帧应用选定的目标检测或跟踪方法。

        参数:
            raw_frame (np.ndarray): 原始图像帧，形状为 (H, W, C)。
            detections (List | None): 可选的目标检测框列表，参与处理。

        返回:
            (np.ndarray): 处理后的帧，或返回运动估计矩阵（如仿射矩阵）。

        示例:
            >>> gmc = GMC(method="sparseOptFlow")
            >>> raw_frame = np.random.rand(480, 640, 3)
            >>> processed_frame = gmc.apply(raw_frame)
            >>> print(processed_frame.shape)
            (480, 640, 3)
        """
        if self.method in {"orb", "sift"}:
            return self.apply_features(raw_frame, detections)
        elif self.method == "ecc":
            return self.apply_ecc(raw_frame)
        elif self.method == "sparseOptFlow":
            return self.apply_sparseoptflow(raw_frame)
        else:
            return np.eye(2, 3)

    def apply_ecc(self, raw_frame: np.array) -> np.array:
        """
        对原始帧应用 ECC（增强相关系数）算法以进行运动补偿。

        参数:
            raw_frame (np.ndarray): 待处理的原始帧，形状为 (H, W, C)。

        返回:
            (np.ndarray): 应用 ECC 变换后的处理帧。

        示例:
            >>> gmc = GMC(method="ecc")
            >>> processed_frame = gmc.apply_ecc(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
            >>> print(processed_frame)
            [[1. 0. 0.]
            [0. 1. 0.]]
        """
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
        H = np.eye(2, 3, dtype=np.float32)  # 初始化仿射变换矩阵为单位矩阵

        # 若需要降采样，则进行图像降采样处理
        if self.downscale > 1.0:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)  # 模糊处理以减少噪声
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))  # 图像尺寸缩小

        # 若是第一帧，则进行初始化
        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()  # 保存当前帧作为参考帧
            self.initializedFirstFrame = True  # 标记已经初始化
            return H  # 返回单位矩阵

        # 运行 ECC 算法，结果保存在变换矩阵 H 中
        # (cc, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria)
        try:
            (_, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria, None, 1)
        except Exception as e:
            LOGGER.warning(f"警告：ECC 变换估计失败，使用单位变换矩阵。异常信息：{e}")

        return H

    def apply_features(self, raw_frame: np.array, detections: list = None) -> np.array:
        """
        对原始帧应用基于特征的方法，如 ORB 或 SIFT。

        参数:
            raw_frame (np.ndarray): 要处理的原始帧，形状为 (H, W, C)。
            detections (List | None): 用于处理的检测框列表。

        返回:
            (np.ndarray): 处理后的帧。

        示例:
            >>> gmc = GMC(method="orb")
            >>> raw_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> processed_frame = gmc.apply_features(raw_frame)
            >>> print(processed_frame.shape)
            (2, 3)
        """
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        # 缩小图像尺寸
        if self.downscale > 1.0:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        # 查找关键点
        mask = np.zeros_like(frame)
        mask[int(0.02 * height): int(0.98 * height), int(0.02 * width): int(0.98 * width)] = 255
        if detections is not None:
            for det in detections:
                tlbr = (det[:4] / self.downscale).astype(np.int_)
                mask[tlbr[1]: tlbr[3], tlbr[0]: tlbr[2]] = 0

        keypoints = self.detector.detect(frame, mask)

        # 计算描述子
        keypoints, descriptors = self.extractor.compute(frame, keypoints)

        # 处理第一帧
        if not self.initializedFirstFrame:
            # 初始化数据
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            # 初始化完成
            self.initializedFirstFrame = True

            return H

        # 进行描述子的匹配
        knnMatches = self.matcher.knnMatch(self.prevDescriptors, descriptors, 2)

        # 根据最小空间距离过滤匹配
        matches = []
        spatialDistances = []

        maxSpatialDistance = 0.25 * np.array([width, height])

        # 处理无匹配的情况
        if len(knnMatches) == 0:
            # 保存到下一帧
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            return H

        for m, n in knnMatches:
            if m.distance < 0.9 * n.distance:
                prevKeyPointLocation = self.prevKeyPoints[m.queryIdx].pt
                currKeyPointLocation = keypoints[m.trainIdx].pt

                spatialDistance = (
                    prevKeyPointLocation[0] - currKeyPointLocation[0],
                    prevKeyPointLocation[1] - currKeyPointLocation[1],
                )

                if (np.abs(spatialDistance[0]) < maxSpatialDistance[0]) and (
                    np.abs(spatialDistance[1]) < maxSpatialDistance[1]
                ):
                    spatialDistances.append(spatialDistance)
                    matches.append(m)

        meanSpatialDistances = np.mean(spatialDistances, 0)
        stdSpatialDistances = np.std(spatialDistances, 0)

        inliers = (spatialDistances - meanSpatialDistances) < 2.5 * stdSpatialDistances

        goodMatches = []
        prevPoints = []
        currPoints = []
        for i in range(len(matches)):
            if inliers[i, 0] and inliers[i, 1]:
                goodMatches.append(matches[i])
                prevPoints.append(self.prevKeyPoints[matches[i].queryIdx].pt)
                currPoints.append(keypoints[matches[i].trainIdx].pt)

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # 在输出图像上绘制关键点匹配
        # if False:
        #     import matplotlib.pyplot as plt
        #     matches_img = np.hstack((self.prevFrame, frame))
        #     matches_img = cv2.cvtColor(matches_img, cv2.COLOR_GRAY2BGR)
        #     W = self.prevFrame.shape[1]
        #     for m in goodMatches:
        #         prev_pt = np.array(self.prevKeyPoints[m.queryIdx].pt, dtype=np.int_)
        #         curr_pt = np.array(keypoints[m.trainIdx].pt, dtype=np.int_)
        #         curr_pt[0] += W
        #         color = np.random.randint(0, 255, 3)
        #         color = (int(color[0]), int(color[1]), int(color[2]))
        #
        #         matches_img = cv2.line(matches_img, prev_pt, curr_pt, tuple(color), 1, cv2.LINE_AA)
        #         matches_img = cv2.circle(matches_img, prev_pt, 2, tuple(color), -1)
        #         matches_img = cv2.circle(matches_img, curr_pt, 2, tuple(color), -1)
        #
        #     plt.figure()
        #     plt.imshow(matches_img)
        #     plt.show()

        # 估计刚性变换矩阵
        if prevPoints.shape[0] > 4:
            H, inliers = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            # 处理缩放还原
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            LOGGER.warning("警告: 匹配点数不足")

        # 保存数据供下一帧使用
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)

        return H

    def apply_sparseoptflow(self, raw_frame: np.array) -> np.array:
        """
        对原始帧应用稀疏光流方法。

        参数:
            raw_frame (np.ndarray): 待处理的原始帧，形状为 (H, W, C)。

        返回:
            (np.ndarray): 处理后的仿射变换矩阵，形状为 (2, 3)。

        示例:
            >>> gmc = GMC()
            >>> result = gmc.apply_sparseoptflow(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
            >>> print(result)
            [[1. 0. 0.]
             [0. 1. 0.]]
        """
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        # 图像下采样
        if self.downscale > 1.0:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        # 计算关键点
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        # 处理第一帧
        if not self.initializedFirstFrame or self.prevKeyPoints is None:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.initializedFirstFrame = True
            return H

        # 计算关键点对应关系（前后帧的光流）
        matchedKeypoints, status, _ = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, self.prevKeyPoints, None)

        # 仅保留匹配成功的关键点对
        prevPoints = []
        currPoints = []

        for i in range(len(status)):
            if status[i]:
                prevPoints.append(self.prevKeyPoints[i])
                currPoints.append(matchedKeypoints[i])

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # 计算刚性变换矩阵（仿射）
        if (prevPoints.shape[0] > 4) and (prevPoints.shape[0] == prevPoints.shape[0]):
            H, _ = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            LOGGER.warning("警告：匹配点数量不足")

        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)

        return H

    def reset_params(self) -> None:
        """重置内部参数，包括上一帧图像、关键点和描述子等。"""
        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False
