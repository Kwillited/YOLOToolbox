import sys
from pathlib import Path
import torch # 用于检测显存
import matplotlib
# 建议在 PyQt6 下使用 QtAgg，但为了兼容您之前的设置，这里保留原样或根据环境调整
# matplotlib.use('QtAgg') 
matplotlib.use('Qt5Agg') 

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (QApplication, QGroupBox, QFormLayout, QLineEdit, QPushButton, QVBoxLayout,
                             QHBoxLayout, QSpinBox, QComboBox, QTextEdit, QLabel, QMessageBox, QWidget, QFileDialog,
                             QSplitter, QSizePolicy) # 新增了 QSplitter
from PyQt6.QtCore import QThread, pyqtSignal, QObject, Qt
from PyQt6.QtGui import QTextCursor
from ultralytics import YOLO
from .utils import StreamRedirector



# --- 核心逻辑：训练线程 ---
class TrainingThread(QThread):
    log_signal = pyqtSignal(str)
    metrics_signal = pyqtSignal(dict)  # 传递结构化数据
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.stop_requested = False

    def run(self):
        # 1. 设置日志重定向
        redirector = StreamRedirector()
        redirector.text_written.connect(self.handle_log)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = redirector
        sys.stderr = redirector

        try:
            self.log_signal.emit(f"🚀 初始化训练...\n模型: {self.params['model']}\n数据: {self.params['data']}\n")
            model = YOLO(self.params['model'])

            # --- 回调函数：获取 Loss 和 mAP 等关键指标 ---
            def on_train_epoch_end(trainer):
                if self.stop_requested:
                    raise InterruptedError("User stopped training")

                current_epoch = trainer.epoch + 1
                
                # 1. 获取显存
                gpu_mem = 0
                if torch.cuda.is_available():
                    # 注意：这里默认取 device 0，如果指定了其他 device 需对应修改
                    try:
                        gpu_mem = torch.cuda.memory_reserved(0) / 1024 / 1024 
                    except:
                        gpu_mem = 0

                # 2. 获取 Loss (Train)
                losses = [0, 0, 0]
                if hasattr(trainer, 'loss_items'):
                    losses = [x.item() for x in trainer.loss_items]

                # 3. 获取 Metrics (Val)
                metrics_dict = trainer.metrics
                map50 = metrics_dict.get('metrics/mAP50(B)', 0)
                map50_95 = metrics_dict.get('metrics/mAP50-95(B)', 0)
                precision = metrics_dict.get('metrics/precision(B)', 0)
                recall = metrics_dict.get('metrics/recall(B)', 0)

                data = {
                    'epoch': current_epoch,
                    'box_loss': losses[0],
                    'cls_loss': losses[1],
                    'dfl_loss': losses[2],
                    'map50': map50,
                    'map50_95': map50_95,
                    'precision': precision,
                    'recall': recall,
                    'gpu_mem': gpu_mem
                }
                
                self.metrics_signal.emit(data)

            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            model.train(
                data=self.params['data'],
                epochs=self.params['epochs'],
                batch=self.params['batch'],
                imgsz=self.params['imgsz'],
                device=self.params['device'],
                workers=2,
                exist_ok=True,
                project=self.params.get('project', 'runs/detect/train')
            )
            self.log_signal.emit("\n✅ 训练完成！结果已保存。")

        except InterruptedError:
            self.log_signal.emit("\n🛑 训练已被用户强制停止。")
        except Exception as e:
            self.error_signal.emit(f"训练出错: {str(e)}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            self.finished_signal.emit()

    def handle_log(self, text):
        self.log_signal.emit(text)

    def stop(self):
        self.stop_requested = True

# --- GUI 模块 ---
class TrainingModule:
    def __init__(self, parent):
        self.parent = parent
        self.train_thread = None
        
        # 图表对象
        self.fig = None
        self.canvas = None
        self.axes = {}
        self.lines = {} 
        
        # 数据缓存
        self.reset_data()
        
    def reset_data(self):
        self.data_cache = {
            'epoch': [], 
            'box_loss': [], 'cls_loss': [], 'dfl_loss': [],
            'map50': [], 'map50_95': [],
            'precision': [], 'recall': [],
            'gpu_mem': []
        }
    
    def detect_available_devices(self):
        """检测系统中可用的设备"""
        devices = []
        device_mapping = {}  # 存储显示名称到实际设备标识的映射
        default_device = "CPU"
        
        # 检测 CUDA GPU
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                display_name = f"GPU {i}"
                actual_device = str(i)
                devices.append(display_name)
                device_mapping[display_name] = actual_device
            if devices:
                default_device = devices[0]  # 默认使用第一个GPU
        
        # 检测 Apple Silicon MPS
        try:
            if torch.backends.mps.is_available():
                display_name = "MPS"
                actual_device = "mps"
                devices.append(display_name)
                device_mapping[display_name] = actual_device
                if default_device == "CPU":  # 如果没有GPU，则默认使用MPS
                    default_device = display_name
        except AttributeError:
            # 如果PyTorch版本不支持MPS，忽略
            pass
        
        # 添加CPU作为备选
        devices.append("CPU")
        device_mapping["CPU"] = "cpu"
        
        self.device_mapping = device_mapping  # 保存映射关系供后续使用
        return devices, default_device

    def init_chart(self):
        """初始化 2x2 图表"""
        # figsize 稍微改小一点，交给 Splitter 管理大小
        self.fig = Figure(figsize=(8, 6), dpi=100) 
        self.canvas = FigureCanvas(self.fig)
        
        # 2x2 布局
        self.axes['loss'] = self.fig.add_subplot(2, 2, 1)
        self.axes['map']  = self.fig.add_subplot(2, 2, 2)
        self.axes['pr']   = self.fig.add_subplot(2, 2, 3)
        self.axes['gpu']  = self.fig.add_subplot(2, 2, 4)
        
        # 1. Loss 图表
        ax_loss = self.axes['loss']
        ax_loss.set_title('Losses', fontsize=10)
        ax_loss.grid(True, alpha=0.3)
        self.lines['box'], = ax_loss.plot([], [], label='Box', color='#1f77b4')
        self.lines['cls'], = ax_loss.plot([], [], label='Cls', color='#ff7f0e')
        self.lines['dfl'], = ax_loss.plot([], [], label='DFL', color='#2ca02c')
        ax_loss.legend(loc='upper right', fontsize='x-small')
        
        # 2. mAP 图表
        ax_map = self.axes['map']
        ax_map.set_title('mAP', fontsize=10)
        ax_map.grid(True, alpha=0.3)
        self.lines['map50'], = ax_map.plot([], [], label='mAP@50', color='#d62728')
        self.lines['map95'], = ax_map.plot([], [], label='mAP@95', color='#9467bd')
        ax_map.legend(loc='lower right', fontsize='x-small')

        # 3. Precision & Recall 图表
        ax_pr = self.axes['pr']
        ax_pr.set_title('P & R', fontsize=10)
        ax_pr.grid(True, alpha=0.3)
        self.lines['precision'], = ax_pr.plot([], [], label='P', color='#8c564b')
        self.lines['recall'], = ax_pr.plot([], [], label='R', color='#e377c2')
        ax_pr.legend(loc='lower right', fontsize='x-small')

        # 4. GPU 图表
        ax_gpu = self.axes['gpu']
        ax_gpu.set_title('GPU (MB)', fontsize=10)
        ax_gpu.grid(True, alpha=0.3)
        self.lines['gpu'], = ax_gpu.plot([], [], label='Mem', color='#7f7f7f', linestyle='--')
        
        self.fig.tight_layout()

    def init_ui(self, tab_train):
        layout = tab_train.layout()
        if not layout:
            layout = QHBoxLayout(tab_train)

        # ---------------- 左侧设置区 (固定宽度) ----------------
        left_widget = self.parent.findChild(QWidget, "train_left_widget")
        if not left_widget:
            left_widget = QWidget()
            left_widget.setObjectName("train_left_widget")
            left_widget.setFixedWidth(320) # 增加宽度以确保按钮显示完整
            layout.addWidget(left_widget)
        
        if left_widget.layout():
            QWidget().setLayout(left_widget.layout()) 
        settings_layout = QVBoxLayout(left_widget)
        settings_layout.setSpacing(10)
        settings_layout.setContentsMargins(0, 0, 5, 0) # 右边留点缝隙

        # 1. 配置
        cfg_group = QGroupBox("训练配置")
        cfg_form = QFormLayout()

        self.data_yaml_edit = QLineEdit("coco128.yaml")
        btn_yaml = QPushButton("...")
        btn_yaml.setFixedWidth(40)  # 增加宽度以确保"..."完整显示
        btn_yaml.clicked.connect(self.select_yaml_file)
        yaml_box = QHBoxLayout()
        yaml_box.addWidget(self.data_yaml_edit)
        yaml_box.addWidget(btn_yaml)

        self.train_model_path = QLineEdit("yolov8n.pt")
        btn_model = QPushButton("...")
        btn_model.setFixedWidth(40)  # 增加宽度以确保"..."完整显示
        btn_model.clicked.connect(self.select_train_base_model)
        model_box = QHBoxLayout()
        model_box.addWidget(self.train_model_path)
        model_box.addWidget(btn_model)

        # 添加训练结果保存路径选择
        self.save_path_edit = QLineEdit("runs/detect/train")
        btn_save_path = QPushButton("...")
        btn_save_path.setFixedWidth(40)
        btn_save_path.clicked.connect(self.select_save_path)
        save_path_box = QHBoxLayout()
        save_path_box.addWidget(self.save_path_edit)
        save_path_box.addWidget(btn_save_path)

        cfg_form.addRow("数据:", yaml_box)
        cfg_form.addRow("模型:", model_box)
        cfg_form.addRow("保存路径:", save_path_box)
        cfg_group.setLayout(cfg_form)

        # 2. 参数
        hyper_group = QGroupBox("超参数")
        hyper_form = QFormLayout()
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 10000)
        self.spin_epochs.setValue(100)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 512)
        self.spin_batch.setValue(16)
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(32, 2048)
        self.spin_imgsz.setValue(640)
        self.spin_imgsz.setSingleStep(32)
        self.combo_device = QComboBox()
        hyper_form.addRow("轮次:", self.spin_epochs)
        hyper_form.addRow("批次:", self.spin_batch)
        hyper_form.addRow("图像尺寸:", self.spin_imgsz)
        hyper_form.addRow("设备:", self.combo_device)
        
        # 检测并设置可用设备
        available_devices, default_device = self.detect_available_devices()
        self.combo_device.addItems(available_devices)
        if default_device in available_devices:
            self.combo_device.setCurrentText(default_device)
        hyper_group.setLayout(hyper_form)

        # 3. 控制
        action_group = QGroupBox("控制")
        action_layout = QVBoxLayout()
        self.btn_start_train = QPushButton("🚀 开始训练")
        self.btn_start_train.setFixedHeight(40)
        self.btn_start_train.clicked.connect(self.start_training)
        self.btn_stop_train = QPushButton("⏹ 停止训练")
        self.btn_stop_train.setEnabled(False)
        self.btn_stop_train.clicked.connect(self.stop_training)
        action_layout.addWidget(self.btn_start_train)
        action_layout.addWidget(self.btn_stop_train)
        action_group.setLayout(action_layout)

        settings_layout.addWidget(cfg_group)
        settings_layout.addWidget(hyper_group)
        settings_layout.addWidget(action_group)
        settings_layout.addStretch()

        # ---------------- 右侧显示区 (使用 QSplitter) ----------------
        right_layout = self.parent.findChild(QVBoxLayout, "train_log_layout")
        if not right_layout:
            right_layout = QVBoxLayout()
            right_layout.setObjectName("train_log_layout")
            layout.addLayout(right_layout)
        else:
            while right_layout.count():
                item = right_layout.takeAt(0)
                if item.widget(): item.widget().deleteLater()

        # 创建分割器 (垂直方向)
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # [Top] 图表区域容器
        chart_widget = QWidget()
        chart_widget.setStyleSheet("border: 2px solid #666; border-radius: 6px; background-color: #222;")
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(10, 10, 10, 10)
        
        self.init_chart()
        chart_title = QLabel("<b>训练指标仪表盘</b>")
        chart_title.setStyleSheet("color: #ffffff; font-size: 12pt; margin-bottom: 5px;")
        chart_layout.addWidget(chart_title)
        chart_layout.addWidget(self.canvas)
        
        # [Bottom] 日志区域容器
        log_widget = QWidget()
        log_widget.setStyleSheet("border: 2px solid #666; border-radius: 6px; background-color: #222;")
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(10, 10, 10, 10)
        
        log_title = QLabel("<b>控制台输出</b>")
        log_title.setStyleSheet("color: #ffffff; font-size: 12pt; margin-bottom: 5px;")
        log_layout.addWidget(log_title)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: Consolas, monospace; font-size: 9pt; background-color: #000000; color: #ffffff; border: 1px solid #444; border-radius: 3px;")
        log_layout.addWidget(self.log_text)

        # 将容器加入分割器
        splitter.addWidget(chart_widget)
        splitter.addWidget(log_widget)
        
        # 设置初始大小比例 [图表高度, 日志高度]
        # 注意：这里是像素值，Splitter 会尝试按此比例分配
        splitter.setSizes([600, 250])

        right_layout.addWidget(splitter)

    def select_yaml_file(self):
        fname, _ = QFileDialog.getOpenFileName(self.parent, '选择YAML配置', '.', "YAML (*.yaml)")
        if fname: self.data_yaml_edit.setText(fname)

    def select_train_base_model(self):
        fname, _ = QFileDialog.getOpenFileName(self.parent, '选择PT权重', '.', "Model (*.pt)")
        if fname: self.train_model_path.setText(fname)

    def select_save_path(self):
        dir_path = QFileDialog.getExistingDirectory(self.parent, "选择保存路径")
        if dir_path: self.save_path_edit.setText(dir_path)

    def start_training(self):
        # 获取用户选择的显示设备名称
        selected_display_device = self.combo_device.currentText()
        # 将显示名称转换为实际设备标识
        actual_device = self.device_mapping.get(selected_display_device, "cpu")
        
        params = {
            'data': self.data_yaml_edit.text(),
            'model': self.train_model_path.text(),
            'epochs': self.spin_epochs.value(),
            'batch': self.spin_batch.value(),
            'imgsz': self.spin_imgsz.value(),
            'device': actual_device,
            'project': self.save_path_edit.text()
        }
        
        # 简单校验
        if not Path(params['data']).exists() and params['data'] != "coco128.yaml":
            QMessageBox.warning(self.parent, "错误", "YAML 数据集文件不存在")
            return

        self.log_text.clear()
        self.btn_start_train.setEnabled(False)
        self.btn_stop_train.setEnabled(True)
        self.reset_data() 
        self.refresh_chart()

        self.train_thread = TrainingThread(params)
        self.train_thread.log_signal.connect(self.append_log)
        self.train_thread.metrics_signal.connect(self.update_data_and_chart)
        self.train_thread.finished_signal.connect(self.training_finished)
        self.train_thread.error_signal.connect(self.training_error)
        self.train_thread.start()

    def stop_training(self):
        if self.train_thread and self.train_thread.isRunning():
            self.btn_stop_train.setEnabled(False)
            self.btn_stop_train.setText("停止中...")
            self.train_thread.stop()

    def append_log(self, text):
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)
        self.log_text.insertPlainText(text)
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)

    def update_data_and_chart(self, metrics):
        self.data_cache['epoch'].append(metrics['epoch'])
        self.data_cache['box_loss'].append(metrics['box_loss'])
        self.data_cache['cls_loss'].append(metrics['cls_loss'])
        self.data_cache['dfl_loss'].append(metrics['dfl_loss'])
        self.data_cache['map50'].append(metrics['map50'])
        self.data_cache['map50_95'].append(metrics['map50_95'])
        self.data_cache['precision'].append(metrics['precision'])
        self.data_cache['recall'].append(metrics['recall'])
        self.data_cache['gpu_mem'].append(metrics['gpu_mem'])
        
        self.refresh_chart()

    def refresh_chart(self):
        epochs = self.data_cache['epoch']
        if not epochs:
            for line in self.lines.values():
                line.set_data([], [])
            self.canvas.draw()
            return

        # 1. Update Losses
        self.lines['box'].set_data(epochs, self.data_cache['box_loss'])
        self.lines['cls'].set_data(epochs, self.data_cache['cls_loss'])
        self.lines['dfl'].set_data(epochs, self.data_cache['dfl_loss'])
        self.axes['loss'].relim()
        self.axes['loss'].autoscale_view()

        # 2. Update mAP
        self.lines['map50'].set_data(epochs, self.data_cache['map50'])
        self.lines['map95'].set_data(epochs, self.data_cache['map50_95'])
        self.axes['map'].relim()
        self.axes['map'].autoscale_view()

        # 3. Update P/R
        self.lines['precision'].set_data(epochs, self.data_cache['precision'])
        self.lines['recall'].set_data(epochs, self.data_cache['recall'])
        self.axes['pr'].relim()
        self.axes['pr'].autoscale_view()

        # 4. Update GPU
        self.lines['gpu'].set_data(epochs, self.data_cache['gpu_mem'])
        self.axes['gpu'].relim()
        self.axes['gpu'].autoscale_view()

        self.canvas.draw()

    def training_finished(self):
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.btn_stop_train.setText("⏹ 停止训练")
        self.log_text.append("\n=== 线程结束 ===")

    def training_error(self, msg):
        QMessageBox.critical(self.parent, "错误", msg)

