import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

# 导入自定义模块
from core.detection import DetectionModule
from core.training import TrainingModule
from core.annotation import AnnotationModule

# ==========================================
# 样式表 (CSS)
# ==========================================
STYLESHEET = """
QMainWindow {
    background-color: #1e1e1e;
}
QLabel {
    color: #e0e0e0;
    font-family: "Segoe UI", "Microsoft YaHei";
    font-size: 14px;
}
/* 分组框 */
QGroupBox {
    border: 1px solid #3d3d3d;
    border-radius: 8px;
    margin-top: 10px;
    color: #00bcd4;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
/* 按钮 */
QPushButton {
    background-color: #007acc;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 8px 15px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #0098ff;
}
QPushButton:pressed {
    background-color: #005c99;
}
QPushButton:disabled {
    background-color: #444444;
    color: #888888;
}
QPushButton#stop_btn {
    background-color: #d32f2f;
}
QPushButton#stop_btn:hover {
    background-color: #f44336;
}
/* 输入控件 */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #2d2d2d;
    color: white;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 5px;
}
/* 滑块 */
QSlider::groove:horizontal {
    border: 1px solid #3d3d3d;
    height: 8px;
    background: #2d2d2d;
    margin: 2px 0;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #007acc;
    border: 1px solid #007acc;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
/* 表格 */
QTableWidget {
    background-color: #252526;
    color: #cccccc;
    gridline-color: #3d3d3d;
    border: 1px solid #3d3d3d;
}
QHeaderView::section {
    background-color: #1e1e1e;
    color: #ffffff;
    padding: 4px;
    border: 1px solid #3d3d3d;
    font-weight: bold;
}
/* 标签页 QTabWidget */
QTabWidget::pane {
    border: 1px solid #3d3d3d;
    background-color: #1e1e1e;
}
QTabBar::tab {
    background: #2d2d2d;
    color: #cccccc;
    padding: 8px 20px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #3d3d3d;
    color: #00bcd4;
    font-weight: bold;
}
QTabBar::tab:hover {
    background: #3e3e42;
}
/* 日志输出框 */
QTextEdit {
    background-color: #000000;
    color: #00ff00;
    font-family: "Consolas", "Courier New";
    border: 1px solid #3d3d3d;
    font-size: 12px;
}

/* 消息框 QMessageBox 基础样式 */
QMessageBox {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: "Segoe UI", "Microsoft YaHei";
    font-size: 14px;
    border: 1px solid #3d3d3d;
    border-radius: 8px;
    padding: 10px;
}

QMessageBox QLabel {
    color: #e0e0e0;
    font-size: 14px;
    margin: 10px 15px;
    padding: 5px;
    min-height: 0px;
    height: auto;
    line-height: 1.5;
}

QMessageBox QPushButton {
    color: white;
    border: none;
    border-radius: 5px;
    padding: 8px 18px;
    font-weight: bold;
    font-size: 13px;
    min-width: 85px;
    margin: 5px 8px;
    transition: background-color 0.2s, border-color 0.2s;
}

QMessageBox QPushButton:hover {
    opacity: 0.9;
}

QMessageBox QPushButton:pressed {
    transform: translateY(1px);
}

/* 消息框标题样式 */
QMessageBox QGroupBox {
    background-color: #1e1e1e;
    border: none;
    font-weight: bold;
    font-size: 16px;
    margin: 5px 10px 15px 10px;
    padding: 5px 10px;
}

/* 消息框按钮容器 */
QMessageBox QDialogButtonBox {
    padding: 5px;
    margin-top: 10px;
    background-color: transparent;
    alignment: center;
}

/* 确保按钮居中对齐 */
QMessageBox QDialogButtonBox QPushButton {
    alignment: center;
}

/* 优化消息框内部布局 */
QMessageBox QVBoxLayout {
    spacing: 10px;
}

QMessageBox QWidget {
    background-color: #1e1e1e;
}

/* 错误类型消息框样式 */
QMessageBox QGroupBox {
    color: #ff4444;
}

/* 信息类型消息框按钮 */
QMessageBox QPushButton {
    background-color: #007acc;
}

QMessageBox QPushButton:hover {
    background-color: #0098ff;
}

QMessageBox QPushButton:pressed {
    background-color: #005c99;
}

/* 确认对话框的特殊按钮样式 */
QMessageBox QPushButton:first-child {
    background-color: #00cc88;
}

QMessageBox QPushButton:first-child:hover {
    background-color: #00eeaa;
}

QMessageBox QPushButton:first-child:pressed {
    background-color: #009966;
}

QMessageBox QPushButton:last-child {
    background-color: #ff4444;
}

QMessageBox QPushButton:last-child:hover {
    background-color: #ff6666;
}

QMessageBox QPushButton:last-child:pressed {
    background-color: #cc0000;
}
"""

class YoloSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO工作台")
        self.setGeometry(100, 100, 1300, 850)
        self.setStyleSheet(STYLESHEET)

        # 初始化模块
        self.detection_module = None
        self.training_module = None
        self.annotation_module = None

        # 主容器
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # 选项卡控件
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # 添加三个标签页
        self.tab_detect = QWidget()
        self.tab_train = QWidget()
        self.tab_annotate = QWidget()

        self.tabs.addTab(self.tab_detect, "🕵️‍♂️ 智能识别")
        self.tabs.addTab(self.tab_train, "🏋️‍♂️ 模型训练")
        self.tabs.addTab(self.tab_annotate, "📝 数据集标注")

        # 初始化各模块
        self.init_modules()

    def init_modules(self):
        """初始化各功能模块"""
        # 初始化检测模块
        self.detection_module = DetectionModule(self)
        self.detection_module.init_ui(self.tab_detect)
        
        # 初始化训练模块
        self.training_module = TrainingModule(self)
        self.training_module.init_ui(self.tab_train)
        
        # 初始化标注模块
        self.annotation_module = AnnotationModule(self)
        self.annotation_module.init_ui(self.tab_annotate)

    def closeEvent(self, event):
        """关闭窗口时的处理"""
        # 停止检测线程
        if hasattr(self.detection_module, 'stop_detection'):
            self.detection_module.stop_detection()
        
        # 停止训练线程
        if hasattr(self.training_module, 'stop'):
            self.training_module.stop()
        
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    window = YoloSystem()
    window.show()
    sys.exit(app.exec())
