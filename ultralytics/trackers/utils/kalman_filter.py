# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np
import scipy.linalg


class KalmanFilterXYAH:
    """
    用于在图像空间中通过卡尔曼滤波器跟踪边界框的 KalmanFilterXYAH 类。

    实现了一个简单的卡尔曼滤波器，用于在图像空间中跟踪边界框。8维状态空间 (x, y, a, h, vx, vy, va, vh)
    包括边界框的中心位置 (x, y)、宽高比 a、高度 h，以及它们各自的速度。
    目标运动遵循匀速模型，边界框的位置 (x, y, a, h) 被视为状态空间的直接观测（线性观测模型）。

    属性:
        _motion_mat (np.ndarray): 卡尔曼滤波器的运动矩阵。
        _update_mat (np.ndarray): 卡尔曼滤波器的观测矩阵。
        _std_weight_position (float): 位置的标准差权重。
        _std_weight_velocity (float): 速度的标准差权重。

    方法:
        initiate: 根据一个未关联的测量初始化一条轨迹。
        predict: 执行卡尔曼滤波器的预测步骤。
        project: 将状态分布投影到观测空间。
        multi_predict: 执行批量预测（向量化版本）。
        update: 执行卡尔曼滤波器的修正步骤。
        gating_distance: 计算状态分布与观测之间的门控距离。

    示例:
        初始化卡尔曼滤波器并从一个测量中创建轨迹
        >>> kf = KalmanFilterXYAH()
        >>> measurement = np.array([100, 200, 1.5, 50])
        >>> mean, covariance = kf.initiate(measurement)
        >>> print(mean)
        >>> print(covariance)
    """

    def __init__(self):
        """
        使用运动与观测不确定性权重初始化卡尔曼滤波器的模型矩阵。

        卡尔曼滤波器使用 8 维状态空间 (x, y, a, h, vx, vy, va, vh)，其中 (x, y) 表示边界框中心，
        'a' 是宽高比，'h' 是高度，其速度部分分别为 (vx, vy, va, vh)。
        滤波器使用匀速运动模型和线性观测模型。

        示例:
            初始化一个卡尔曼滤波器用于跟踪：
            >>> kf = KalmanFilterXYAH()
        """
        ndim, dt = 4, 1.0

        # 创建卡尔曼滤波器的模型矩阵
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # 运动和观测的不确定性相对于当前状态估计选择，这些权重控制模型中不确定度的大小
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple:
        """
        从一个未关联的测量值创建一条新轨迹。

        参数:
            measurement (ndarray): 边界框坐标 (x, y, a, h)，其中 (x, y) 为中心位置，a 为宽高比，h 为高度。

        返回:
            (tuple[ndarray, ndarray]): 返回新的轨迹的均值向量（8 维）和协方差矩阵（8x8）。
                未观测的速度部分初始为均值 0。

        示例:
            >>> kf = KalmanFilterXYAH()
            >>> measurement = np.array([100, 50, 1.5, 200])
            >>> mean, covariance = kf.initiate(measurement)
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        执行卡尔曼滤波器的预测步骤。

        参数:
            mean (ndarray): 上一时刻的状态均值向量（8 维）。
            covariance (ndarray): 上一时刻的状态协方差矩阵（8x8）。

        返回:
            (tuple[ndarray, ndarray]): 返回预测后的状态均值向量与协方差矩阵。

        示例:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> predicted_mean, predicted_covariance = kf.predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        将状态分布投影到观测空间。

        参数:
            mean (ndarray): 当前状态的均值向量（8 维）。
            covariance (ndarray): 当前状态的协方差矩阵（8x8）。

        返回:
            (tuple[ndarray, ndarray]): 返回该状态在观测空间中的投影后的均值和协方差矩阵。

        示例:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> projected_mean, projected_covariance = kf.project(mean, covariance)
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        对多个目标状态运行 Kalman 滤波器的预测步骤（向量化版本）。

        参数:
            mean (ndarray): 上一时刻各目标状态的均值矩阵，维度为 Nx8。
            covariance (ndarray): 上一时刻各目标状态的协方差矩阵，维度为 Nx8x8。

        返回:
            (tuple[ndarray, ndarray]): 返回预测后的状态均值矩阵和协方差矩阵。
                均值矩阵形状为 (N, 8)，协方差矩阵形状为 (N, 8, 8)。未观测到的速度分量初始化为 0 均值。

        示例:
            >>> mean = np.random.rand(10, 8)  # 10 个目标状态
            >>> covariance = np.random.rand(10, 8, 8)  # 每个目标的协方差矩阵
            >>> predicted_mean, predicted_covariance = kalman_filter.multi_predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple:
        """
        运行 Kalman 滤波器的校正步骤。

        参数:
            mean (ndarray): 预测状态的均值向量（8 维）。
            covariance (ndarray): 状态的协方差矩阵（8x8）。
            measurement (ndarray): 4 维测量向量 (x, y, a, h)，其中 (x, y) 是中心位置，a 是宽高比，h 是目标高度。

        返回:
            (tuple[ndarray, ndarray]): 返回校正后的状态分布（均值和协方差）。

        示例:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> measurement = np.array([1, 1, 1, 1])
            >>> new_mean, new_covariance = kf.update(mean, covariance, measurement)
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        """
        计算状态分布与测量之间的门控距离（gating distance）。

        一个合适的距离阈值可以通过 `chi2inv95` 获取。如果 `only_position` 为 False，卡方分布的自由度为 4，
        否则为 2。

        参数:
            mean (ndarray): 状态分布的均值向量（8 维）。
            covariance (ndarray): 状态分布的协方差矩阵（8x8）。
            measurements (ndarray): 一个 (N, 4) 的矩阵，表示 N 个测量值，每个测量为 (x, y, a, h)，
                                    其中 (x, y) 是中心位置，a 是宽高比，h 是高度。
            only_position (bool): 如果为 True，仅使用位置 (x, y) 进行距离计算。
            metric (str): 距离度量方式，"gaussian" 表示平方欧式距离，"maha" 表示平方马氏距离。

        返回:
            (np.ndarray): 返回一个长度为 N 的数组，第 i 个元素表示状态分布与 `measurements[i]` 之间的平方距离。

        示例:
            使用马氏距离计算门控距离：
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> measurements = np.array([[1, 1, 1, 1], [2, 2, 1, 1]])
            >>> distances = kf.gating_distance(mean, covariance, measurements, only_position=False, metric="maha")
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return np.sum(z * z, axis=0)  # 马氏距离的平方
        else:
            raise ValueError("无效的距离度量方式")


class KalmanFilterXYWH(KalmanFilterXYAH):
    """
    一个用于图像空间目标框跟踪的 KalmanFilterXYWH 类。

    实现了基于 Kalman 滤波的目标框跟踪器，其状态空间为 (x, y, w, h, vx, vy, vw, vh)，
    其中 (x, y) 为中心位置，w 为宽度，h 为高度，vx, vy, vw, vh 分别是位置和尺寸的速度分量。
    目标的运动遵循恒速模型，边界框的位置 (x, y, w, h) 被视为状态空间的直接观测（线性观测模型）。

    属性:
        _motion_mat (np.ndarray): Kalman 滤波器的运动矩阵。
        _update_mat (np.ndarray): Kalman 滤波器的观测更新矩阵。
        _std_weight_position (float): 位置的标准差权重。
        _std_weight_velocity (float): 速度的标准差权重。

    方法:
        initiate: 根据未关联的测量值创建跟踪轨迹。
        predict: 执行 Kalman 滤波的预测步骤。
        project: 将状态分布投影到观测空间。
        multi_predict: 向量化地执行 Kalman 滤波的预测步骤。
        update: 执行 Kalman 滤波的校正步骤。

    示例:
        创建 Kalman 滤波器并初始化一条跟踪轨迹：
        >>> kf = KalmanFilterXYWH()
        >>> measurement = np.array([100, 50, 20, 40])
        >>> mean, covariance = kf.initiate(measurement)
        >>> print(mean)
        >>> print(covariance)
    """

    def initiate(self, measurement: np.ndarray) -> tuple:
        """
        从未关联的观测值初始化一个跟踪目标。

        参数:
            measurement (ndarray): 边界框坐标 (x, y, w, h)，其中 (x, y) 为中心点坐标，w 为宽度，h 为高度。

        返回:
            (tuple[ndarray, ndarray]): 返回新的跟踪目标的均值向量（8维）和协方差矩阵（8x8维）。
                未观测到的速度部分初始化为均值0。

        示例:
            >>> kf = KalmanFilterXYWH()
            >>> measurement = np.array([100, 50, 20, 40])
            >>> mean, covariance = kf.initiate(measurement)
            >>> print(mean)
            [100.  50.  20.  40.   0.   0.   0.   0.]
            >>> print(covariance)
            [[ 4.  0.  0.  0.  0.  0.  0.  0.]
            [ 0.  4.  0.  0.  0.  0.  0.  0.]
            [ 0.  0.  4.  0.  0.  0.  0.  0.]
            [ 0.  0.  0.  4.  0.  0.  0.  0.]
            [ 0.  0.  0.  0.  0.25  0.  0.  0.]
            [ 0.  0.  0.  0.  0.  0.25  0.  0.]
            [ 0.  0.  0.  0.  0.  0.  0.25  0.]
            [ 0.  0.  0.  0.  0.  0.  0.  0.25]]
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance) -> tuple:
        """
        执行卡尔曼滤波器的预测步骤。

        参数:
            mean (ndarray): 上一时刻对象状态的8维均值向量。
            covariance (ndarray): 上一时刻对象状态的8x8协方差矩阵。

        返回:
            (tuple[ndarray, ndarray]): 返回预测后的状态的均值向量和协方差矩阵。
                未观测的速度部分仍为均值0。

        示例:
            >>> kf = KalmanFilterXYWH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> predicted_mean, predicted_covariance = kf.predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance) -> tuple:
        """
        将状态分布投影到观测空间。

        参数:
            mean (ndarray): 状态的均值向量（8维数组）。
            covariance (ndarray): 状态的协方差矩阵（8x8维）。

        返回:
            (tuple[ndarray, ndarray]): 返回该状态估计在观测空间中的投影均值和协方差矩阵。

        示例:
            >>> kf = KalmanFilterXYWH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> projected_mean, projected_cov = kf.project(mean, covariance)
        """
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance) -> tuple:
        """
        执行卡尔曼滤波器的预测步骤（向量化版本，支持批量处理）。

        参数:
            mean (ndarray): 对象状态的 Nx8 维均值矩阵，N 是对象个数。
            covariance (ndarray): 对象状态的 Nx8x8 协方差矩阵。

        返回:
            (tuple[ndarray, ndarray]): 返回预测后的状态均值向量和协方差矩阵。
                未观测的速度部分仍为均值0。

        示例:
            >>> mean = np.random.rand(5, 8)  # 5个对象的状态向量（8维）
            >>> covariance = np.random.rand(5, 8, 8)  # 5个对象的协方差矩阵（8x8）
            >>> kf = KalmanFilterXYWH()
            >>> predicted_mean, predicted_covariance = kf.multi_predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement) -> tuple:
        """
        执行卡尔曼滤波的校正（更新）步骤。

        参数:
            mean (ndarray): 预测状态的均值向量（8维）。
            covariance (ndarray): 状态的协方差矩阵（8x8维）。
            measurement (ndarray): 4维的观测向量 (x, y, w, h)，其中 (x, y) 是目标中心位置，w 是宽度，h 是高度。

        返回:
            (tuple[ndarray, ndarray]): 返回经过观测修正后的状态分布（均值和协方差）。

        示例:
            >>> kf = KalmanFilterXYWH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> measurement = np.array([0.5, 0.5, 1.2, 1.2])
            >>> new_mean, new_covariance = kf.update(mean, covariance, measurement)
        """
        return super().update(mean, covariance, measurement)
