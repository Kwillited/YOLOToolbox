import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, QMutex, Qt
from PyQt6.QtGui import QImage, QPixmap

# 用于捕获 print 输出并发送信号
class StreamRedirector(QObject):
    text_written = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.buffer = ''
        self.mutex = QMutex()  # 防止多线程访问冲突

    def write(self, text):
        self.mutex.lock()
        try:
            # 处理不同类型的输入
            if isinstance(text, bytes):
                # 尝试不同编码解码
                try:
                    text = text.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        text = text.decode('gbk')
                    except UnicodeDecodeError:
                        text = text.decode('latin-1', errors='replace')
            elif not isinstance(text, str):
                text = str(text)

            # 累积文本
            self.buffer += text
            
            # 当遇到换行符或缓冲区过大时发送信号
            if '\n' in text or len(self.buffer) > 1000:
                self.text_written.emit(self.buffer)
                self.buffer = ''
        finally:
            self.mutex.unlock()

    def flush(self):
        self.mutex.lock()
        try:
            if self.buffer:
                self.text_written.emit(self.buffer)
                self.buffer = ''
        finally:
            self.mutex.unlock()


def cv_img_to_qt(cv_img):
    """
    将 OpenCV 图像转换为 QPixmap
    [重要修复]：增加了 .copy() 以确保内存安全
    """
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    # 必须使用 .copy()，否则 rgb_image 被回收后 QImage 会指向垃圾内存导致崩溃
    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
    
    # 直接返回原始尺寸的 Pixmap，确保图片完整显示
    return QPixmap.fromImage(qt_image)