# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator


class AIGym(BaseSolution):
    """
    一个用于在实时视频流中根据人体姿态管理健身动作的类。

    该类继承自 BaseSolution，利用 YOLO 姿态估计模型来监测锻炼动作。它根据预定义的“上”与“下”姿态角度阈值，
    对动作进行跟踪和计数。

    属性:
        count (List[int]): 每个检测到的人物的重复次数计数。
        angle (List[float]): 每个人物当前跟踪身体部位的角度。
        stage (List[str]): 每个人当前的锻炼阶段（'up', 'down', 或 '-'）。
        initial_stage (str | None): 锻炼初始阶段。
        up_angle (float): 判断“上”姿态所需的角度阈值。
        down_angle (float): 判断“下”姿态所需的角度阈值。
        kpts (List[int]): 用于角度计算的关键点索引。
        annotator (Annotator): 用于在图像上绘制注释的对象。

    方法:
        monitor: 处理帧，检测姿态、计算角度并统计重复次数。

    示例:
        >>> gym = AIGym(model="yolov8n-pose.pt")
        >>> image = cv2.imread("gym_scene.jpg")
        >>> processed_image = gym.monitor(image)
        >>> cv2.imshow("Processed Image", processed_image)
        >>> cv2.waitKey(0)
    """

    def __init__(self, **kwargs):
        """使用姿态估计和预定义角度初始化 AIGym 用于锻炼监测。"""
        # 检查模型名称是否以 '-pose' 结尾
        if "model" in kwargs and "-pose" not in kwargs["model"]:
            kwargs["model"] = "yolo11n-pose.pt"
        elif "model" not in kwargs:
            kwargs["model"] = "yolo11n-pose.pt"

        super().__init__(**kwargs)
        self.count = []  # 用于计数的列表，适用于画面中出现多个人物的情况
        self.angle = []  # 用于存储角度的列表，适用于画面中出现多个人物的情况
        self.stage = []  # 用于存储阶段的列表，适用于画面中出现多个人物的情况

        # 从 CFG 中一次性提取参数，后续复用
        self.initial_stage = None
        self.up_angle = float(self.CFG["up_angle"])  # 判断“上”姿态的角度阈值
        self.down_angle = float(self.CFG["down_angle"])  # 判断“下”姿态的角度阈值
        self.kpts = self.CFG["kpts"]  # 用户定义的用于计算动作的关键点索引

    def monitor(self, im0):
        """
        使用 Ultralytics YOLO 姿态模型监测锻炼动作。

        本函数处理输入图像，跟踪并分析人体姿态，以监控锻炼动作。
        它使用 YOLO Pose 模型来检测关键点，估算角度，并根据预设阈值计算重复次数。

        参数:
            im0 (ndarray): 要处理的输入图像。

        返回:
            (ndarray): 添加了锻炼监控注释的处理后图像。

        示例:
            >>> gym = AIGym()
            >>> image = cv2.imread("workout.jpg")
            >>> processed_image = gym.monitor(image)
        """
        # 获取追踪数据
        tracks = self.model.track(source=im0, persist=True, classes=self.CFG["classes"], **self.track_add_args)[0]

        if tracks.boxes.id is not None:
            # 提取并检查关键点
            if len(tracks) > len(self.count):
                new_human = len(tracks) - len(self.count)
                self.angle += [0] * new_human
                self.count += [0] * new_human
                self.stage += ["-"] * new_human

            # 初始化注释器
            self.annotator = Annotator(im0, line_width=self.line_width)

            # 遍历关键点
            for ind, k in enumerate(reversed(tracks.keypoints.data)):
                # 获取关键点并计算角度
                kpts = [k[int(self.kpts[i])].cpu() for i in range(3)]
                self.angle[ind] = self.annotator.estimate_pose_angle(*kpts)
                im0 = self.annotator.draw_specific_points(k, self.kpts, radius=self.line_width * 3)

                # 根据角度阈值判断当前阶段和更新计数
                if self.angle[ind] < self.down_angle:
                    if self.stage[ind] == "up":
                        self.count[ind] += 1
                    self.stage[ind] = "down"
                elif self.angle[ind] > self.up_angle:
                    self.stage[ind] = "up"

                # 显示角度、计数和阶段文字
                self.annotator.plot_angle_and_count_and_stage(
                    angle_text=self.angle[ind],  # 用于显示的角度
                    count_text=self.count[ind],  # 锻炼动作的计数
                    stage_text=self.stage[ind],  # 当前的姿态阶段
                    center_kpt=k[int(self.kpts[1])],  # 用于显示文字的中心关键点
                )

        self.display_output(im0)  # 显示图像（如果环境支持显示）
        return im0  # 返回处理后的图像用于保存或进一步使用
