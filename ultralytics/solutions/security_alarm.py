# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors


class SecurityAlarm(BaseSolution):
    """
    用于实时监控的安全告警管理类。

    该类继承自 BaseSolution，提供了在图像中监测目标数量、
    当检测总量超过设定阈值时发送邮件通知，并在输出图像上进行可视化注释的功能。

    属性：
       email_sent (bool)：用于记录当前事件是否已发送过邮件的标志。
       records (int)：触发告警所需的检测目标数量阈值。

    方法：
       authenticate：设置邮件服务器认证，用于发送警报。
       send_email：发送带有检测详情与图像附件的邮件通知。
       monitor：监测帧内容，处理检测信息，并在超过阈值时触发警报。

    示例：
        >>> security = SecurityAlarm()
        >>> security.authenticate("abc@gmail.com", "1111222233334444", "xyz@gmail.com")
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = security.monitor(frame)
    """

    def __init__(self, **kwargs):
        """初始化 SecurityAlarm 类，用于实时目标监控。"""
        super().__init__(**kwargs)
        self.email_sent = False
        self.records = self.CFG["records"]
        self.server = None
        self.to_email = ""
        self.from_email = ""

    def authenticate(self, from_email, password, to_email):
        """
        验证邮件服务器身份信息，用于发送告警邮件。

        参数：
            from_email (str)：发件人邮箱地址。
            password (str)：发件人邮箱的登录密码。
            to_email (str)：收件人邮箱地址。

        此方法会初始化与 SMTP 邮件服务器的安全连接，并使用提供的账号信息登录。

        示例：
            >>> alarm = SecurityAlarm()
            >>> alarm.authenticate("sender@example.com", "password123", "recipient@example.com")
        """
        import smtplib

        self.server = smtplib.SMTP("smtp.gmail.com: 587")
        self.server.starttls()
        self.server.login(from_email, password)
        self.to_email = to_email
        self.from_email = from_email

    def send_email(self, im0, records=5):
        """
        发送一封带有图像附件的警报邮件，说明检测到的目标数量。

        参数：
            im0 (numpy.ndarray)：要附加到邮件中的输入图像或帧。
            records (int)：检测到的目标数量，将包含在邮件内容中。

        此方法会将输入图像编码，组合邮件正文与图像附件，并发送给指定收件人。

        示例：
            >>> alarm = SecurityAlarm()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> alarm.send_email(frame, records=10)
        """
        from email.mime.image import MIMEImage
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        import cv2

        img_bytes = cv2.imencode(".jpg", im0)[1].tobytes()  # 将图像编码为 JPEG 格式

        # 创建邮件
        message = MIMEMultipart()
        message["From"] = self.from_email
        message["To"] = self.to_email
        message["Subject"] = "Security Alert"

        # 添加邮件正文
        message_body = f"Ultralytics ALERT!!! {records} objects have been detected!!"
        message.attach(MIMEText(message_body))

        # 添加图像附件
        image_attachment = MIMEImage(img_bytes, name="ultralytics.jpg")
        message.attach(image_attachment)

        # 发送邮件
        try:
            self.server.send_message(message)
            LOGGER.info("✅ 邮件发送成功！")
        except Exception as e:
            print(f"❌ 邮件发送失败: {e}")

    def monitor(self, im0):
        """
        监测帧内容，处理目标检测，并在超过阈值时触发警报。

        参数：
            im0 (numpy.ndarray)：待处理并注释的输入图像或帧。

        此方法会处理输入帧，提取检测信息，为图像添加边框注释，
        如果检测到的目标数量超过设定阈值，且尚未发送邮件，则会发送警报通知。

        返回：
            (numpy.ndarray)：已添加注释的处理图像。

        示例：
            >>> alarm = SecurityAlarm()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> processed_frame = alarm.monitor(frame)
        """
        self.annotator = Annotator(im0, line_width=self.line_width)  # 初始化标注工具
        self.extract_tracks(im0)  # 提取目标轨迹

        # 遍历所有目标框与类别索引
        for box, cls in zip(self.boxes, self.clss):
            # 绘制边框
            self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))

        total_det = len(self.clss)
        if total_det > self.records and not self.email_sent:  # 仅在未发送过邮件时才发送
            self.send_email(im0, total_det)
            self.email_sent = True

        self.display_output(im0)  # 使用基类方法显示输出

        return im0  # 返回处理后的图像供进一步使用
