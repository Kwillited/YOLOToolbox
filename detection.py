import sys
import time
import cv2
import numpy as np
from pathlib import Path
from mss import mss

from PyQt6.QtWidgets import (QApplication, QLabel, QComboBox, QSlider, 
                             QPushButton, QGroupBox, QVBoxLayout, QHBoxLayout, QFormLayout, 
                             QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, 
                             QFileDialog, QMessageBox, QWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor, QCursor
from ultralytics import YOLO
from .utils import cv_img_to_qt

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, list)

    def __init__(self, source_type='camera'):
        super().__init__()
        self._run_flag = True
        self.model = None
        self.conf = 0.25
        self.iou = 0.45
        self.source = 0
        self.source_type = source_type  # 'camera' or 'screen'
        self.monitor_index = 1          # 默认抓取主屏

    def set_model(self, model_or_path):
        """支持传入路径字符串或已加载的模型对象"""
        if isinstance(model_or_path, str):
            self.model = YOLO(model_or_path)
        else:
            self.model = model_or_path

    def set_params(self, conf, iou):
        self.conf = conf
        self.iou = iou

    def set_monitor(self, index):
        """设置要抓取的屏幕索引"""
        self.monitor_index = index

    def run(self):
        # --- 屏幕捕获模式 ---
        if self.source_type == 'screen':
            with mss() as sct:
                # 校验显示器索引，防止越界
                try:
                    # 如果索引超出范围，回退到主屏(1)或全屏(0)
                    if self.monitor_index >= len(sct.monitors):
                        target_mon_idx = 1
                    else:
                        target_mon_idx = self.monitor_index
                    
                    monitor = sct.monitors[target_mon_idx]
                except Exception:
                    # 极端情况兜底
                    monitor = sct.monitors[0]

                while self._run_flag:
                    start_time = time.time()
                    
                    # 截图并转换
                    try:
                        screenshot = sct.grab(monitor)
                        frame = np.array(screenshot)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        
                        self._process_and_emit(frame)
                    except Exception as e:
                        print(f"Screen capture error: {e}")
                    
                    # FPS 控制 (限制在 ~30 FPS，减少CPU占用)
                    self._cap_fps(start_time)

        # --- 摄像头模式 ---
        elif self.source_type == 'camera':
            cap = cv2.VideoCapture(self.source)
            while self._run_flag:
                start_time = time.time()
                ret, frame = cap.read()
                if ret:
                    self._process_and_emit(frame)
                else:
                    # 如果摄像头读取失败（如被占用），稍微等待避免死循环
                    time.sleep(0.1)
                
                self._cap_fps(start_time)
            cap.release()

    def _process_and_emit(self, frame):
        """统一的推理和信号发送逻辑"""
        if self.model:
            # verbose=False 防止控制台刷屏
            results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
            annotated_frame = results[0].plot()
            detections = []
            # 解析结果
            if results[0].boxes:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id] if self.model.names else str(cls_id)
                    detections.append({
                        "class": class_name,
                        "conf": float(box.conf[0]),
                        "box": box.xyxy[0].tolist()
                    })
            self.change_pixmap_signal.emit(annotated_frame, detections)
        else:
            self.change_pixmap_signal.emit(frame, [])

    def _cap_fps(self, start_time):
        """控制帧率，释放CPU"""
        elapsed = time.time() - start_time
        target_delay = 0.033  # 约 30 FPS
        if elapsed < target_delay:
            time.sleep(target_delay - elapsed)

    def stop(self):
        self._run_flag = False
        self.wait()


class VideoPlayerThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, list)
    playback_finished_signal = pyqtSignal()
    
    def __init__(self, video_path, model):
        super().__init__()
        self._run_flag = True
        self._pause_flag = False
        self.video_path = video_path
        self.model = model
        self.conf = 0.25
        self.iou = 0.45
        self.speed = 1.0  # 播放速度倍数
        self.current_frame = 0
        self.total_frames = 0
    
    def set_params(self, conf, iou):
        self.conf = conf
        self.iou = iou
    
    def set_speed(self, speed):
        self.speed = max(0.1, min(3.0, speed))  # 限制速度在0.1x到3.0x之间
    
    def pause(self):
        self._pause_flag = True
    
    def resume(self):
        self._pause_flag = False
    
    def toggle_pause(self):
        self._pause_flag = not self._pause_flag
    
    def seek(self, frame_number):
        self.current_frame = max(0, min(self.total_frames - 1, frame_number))
    
    def fast_forward(self, seconds=5):
        # 快进指定秒数
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30
        cap.release()
        self.seek(self.current_frame + int(seconds * fps * self.speed))
    
    def rewind(self, seconds=5):
        # 后退指定秒数
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30
        cap.release()
        self.seek(self.current_frame - int(seconds * fps * self.speed))
    
    def stop(self):
        self._run_flag = False
        self.wait()
    
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.playback_finished_signal.emit()
            return
        
        # 获取视频属性
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30  # 默认30fps
        
        while self._run_flag:
            # 暂停功能
            while self._pause_flag and self._run_flag:
                time.sleep(0.1)
                continue
            
            # 设置当前播放位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            
            ret, frame = cap.read()
            if not ret:
                break  # 视频播放完毕
            
            # 更新当前帧计数
            self.current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # 执行目标检测
            results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
            annotated_frame = results[0].plot()
            detections = []
            if results[0].boxes:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id] if self.model.names else str(cls_id)
                    detections.append({
                        "class": class_name,
                        "conf": float(box.conf[0]),
                        "box": box.xyxy[0].tolist()
                    })
            
            # 发送信号更新UI
            self.change_pixmap_signal.emit(annotated_frame, detections)
            
            # 控制播放速度
            frame_delay = int(1000 / (fps * self.speed))
            time.sleep(frame_delay / 1000.0)
        
        cap.release()
        self.playback_finished_signal.emit()


class DetectionModule:
    def __init__(self, parent):
        self.parent = parent
        self.model_path = "yolov8n.pt"
        self.model = None  # 初始化为 None，稍后加载
        self.video_thread = None
        self.video_player_thread = None
        self.current_file = None  # 存储当前选择的文件路径
        self.is_running = False  # 运行状态标志，用于控制视频处理循环
        
        # 用于存储检测结果
        self.latest_frame = None  # 最新的检测帧（用于保存）
        self.latest_detections = []  # 最新的检测结果数据
        self.current_file_type = None  # 当前处理的文件类型：'image' 或 'video'
        
        # 预加载默认模型（可选，避免第一次卡顿）
        try:
            self.model = YOLO(self.model_path)
        except Exception:
            print(f"提示: 默认模型 {self.model_path} 未找到，请在界面选择或下载。")

    def init_ui(self, tab_detect):
        layout = tab_detect.layout()
        if not layout:
            layout = QHBoxLayout(tab_detect)

        # --- 左侧控制面板 ---
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(10)

        # 1. 识别模式设置
        mode_group = QGroupBox("识别模式")
        mode_layout = QVBoxLayout()
        self.detect_mode_combo = QComboBox()
        self.detect_mode_combo.addItems(["📂 图片/视频识别", "📹 摄像头识别", "🖥️ 桌面识别"])
        # 连接信号到槽函数，用于控制输入源设置的可见性
        self.detect_mode_combo.currentIndexChanged.connect(self.on_detect_mode_changed)
        mode_layout.addWidget(QLabel("识别模式:"))
        mode_layout.addWidget(self.detect_mode_combo)
        mode_group.setLayout(mode_layout)

        # 2. 输入源设置 (新增：解决无限放大问题)
        self.screen_group = QGroupBox("输入源设置")
        screen_layout = QVBoxLayout()
        self.monitor_combo = QComboBox()
        # 获取屏幕列表
        try:
            with mss() as sct:
                for i, m in enumerate(sct.monitors):
                    if i == 0:
                        self.monitor_combo.addItem(f"全屏拼接 (Index:0)")
                    else:
                        self.monitor_combo.addItem(f"显示器 {i}: {m['width']}x{m['height']}")
            # 默认尝试选中第二个选项（通常是主显示器 Index 1）
            if self.monitor_combo.count() > 1:
                self.monitor_combo.setCurrentIndex(1)
        except Exception as e:
            self.monitor_combo.addItem("无法检测屏幕")
            print(f"Monitor detect error: {e}")

        screen_layout.addWidget(QLabel("选择截取屏幕:"))
        screen_layout.addWidget(self.monitor_combo)
        self.screen_group.setLayout(screen_layout)
        
        # 默认隐藏输入源设置，只有在选择桌面识别时才显示
        self.screen_group.hide()

        # 3. 模型设置
        model_group = QGroupBox("推理模型")
        model_layout = QVBoxLayout()
        self.det_model_combo = QComboBox()
        self.det_model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "自定义..."])
        self.det_model_combo.currentTextChanged.connect(self.select_detect_model)
        model_layout.addWidget(QLabel("选择模型:"))
        model_layout.addWidget(self.det_model_combo)
        model_group.setLayout(model_layout)

        # 3. 参数调整
        param_group = QGroupBox("参数调整")
        param_layout = QFormLayout()
        
        self.conf_label = QLabel("置信度: 0.25")
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(self.update_detect_params)
        
        self.iou_label = QLabel("IoU 阈值: 0.45")
        self.iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.iou_slider.setRange(1, 99)
        self.iou_slider.setValue(45)
        self.iou_slider.valueChanged.connect(self.update_detect_params)
        
        param_layout.addRow(self.conf_label, self.conf_slider)
        param_layout.addRow(self.iou_label, self.iou_slider)
        param_group.setLayout(param_layout)

        # 5. 功能按钮
        btn_group = QGroupBox("功能控制")
        btn_layout = QVBoxLayout()
        
        # 选择识别文件按钮
        self.btn_select_file = QPushButton("📁 选择识别文件")
        self.btn_select_file.clicked.connect(self.open_image)
        
        # 开始和停止按钮
        self.btn_start = QPushButton("▶ 开始识别")
        self.btn_start.clicked.connect(self.start_detection)
        self.btn_stop = QPushButton("⏹ 停止识别")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_detection)
        
        # 保存结果按钮
        self.btn_save_result = QPushButton("💾 保存结果")
        self.btn_save_result.clicked.connect(self.save_detection_result)
        self.btn_save_result.setEnabled(False)  # 初始禁用，检测完成后启用
        
        btn_layout.addWidget(self.btn_select_file)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_save_result)
        
        # 视频播放控制按钮（初始隐藏）
        video_controls_group = QGroupBox("视频控制")
        video_controls_layout = QVBoxLayout()
        
        # 播放控制按钮行
        playback_controls = QHBoxLayout()
        self.btn_rewind = QPushButton("⏪ 后退")
        self.btn_rewind.clicked.connect(self.rewind_video)
        self.btn_play_pause = QPushButton("⏯ 暂停")
        self.btn_play_pause.clicked.connect(self.toggle_video_pause)
        self.btn_forward = QPushButton("⏩ 快进")
        self.btn_forward.clicked.connect(self.forward_video)
        
        playback_controls.addWidget(self.btn_rewind)
        playback_controls.addWidget(self.btn_play_pause)
        playback_controls.addWidget(self.btn_forward)
        video_controls_layout.addLayout(playback_controls)
        
        # 播放速度控制
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("播放速度:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 30)  # 0.1x到3.0x
        self.speed_slider.setValue(10)  # 默认1.0x
        self.speed_slider.valueChanged.connect(self.change_video_speed)
        self.speed_label = QLabel("1.0x")
        
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_label)
        video_controls_layout.addLayout(speed_layout)
        
        video_controls_group.setLayout(video_controls_layout)
        video_controls_group.hide()  # 初始隐藏
        
        # 将视频控制组添加到功能按钮布局
        btn_layout.addWidget(video_controls_group)
        
        # 保存视频控制组引用
        self.video_controls_group = video_controls_group
        
        btn_group.setLayout(btn_layout)

        # 6. 数据表格
        data_group = QGroupBox("实时结果")
        data_layout = QVBoxLayout()
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(3)
        self.result_table.setHorizontalHeaderLabels(["类别", "置信度", "坐标"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        data_layout.addWidget(self.result_table)
        data_group.setLayout(data_layout)

        controls_layout.addWidget(mode_group)
        controls_layout.addWidget(self.screen_group)
        controls_layout.addWidget(model_group)
        controls_layout.addWidget(param_group)
        controls_layout.addWidget(btn_group)
        controls_layout.addWidget(data_group)

        # 将左侧面板添加到主布局
        left_widget = QWidget()
        left_widget.setLayout(controls_layout)
        left_widget.setFixedWidth(320)
        layout.addWidget(left_widget)

        # --- 右侧显示区域 ---
        display_layout = QVBoxLayout()
        self.image_label = QLabel("等待输入...\n(若桌面识别无限放大，请在左上角切换屏幕)")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #000; border: 2px solid #333; border-radius: 5px; color: #888;")
        display_layout.addWidget(self.image_label)
        layout.addLayout(display_layout)

    def select_detect_model(self, text):
        """切换模型逻辑"""
        if text == "自定义...":
            fname, _ = QFileDialog.getOpenFileName(self.parent, '选择模型', '.', "YOLO Model (*.pt)")
            if fname:
                # 将路径保存到 itemData，显示只显示文件名
                display_name = Path(fname).name
                
                # 检查是否已经存在相同文件名的模型
                existing_idx = -1
                for i in range(self.det_model_combo.count()):
                    if i < 2:  # 跳过前两个默认模型
                        continue
                    if self.det_model_combo.itemText(i) == display_name:
                        existing_idx = i
                        break
                
                if existing_idx != -1:
                    # 如果已存在，直接选中
                    self.det_model_combo.setCurrentIndex(existing_idx)
                else:
                    # 如果不存在，添加新选项
                    self.det_model_combo.addItem(display_name, fname)
                    self.det_model_combo.setCurrentIndex(self.det_model_combo.count() - 1)
                
                # 立即加载自定义模型，避免后续路径问题
                self.model_path = fname
                
                # 加载模型 (显示忙碌光标)
                QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
                try:
                    print(f"Loading model: {self.model_path}")
                    self.model = YOLO(self.model_path)
                    # 如果线程正在运行，实时更新线程中的模型
                    if self.video_thread and self.video_thread.isRunning():
                        self.video_thread.set_model(self.model)
                except Exception as e:
                    QMessageBox.critical(self.parent, "错误", f"模型加载失败: {e}\n请确认文件存在且格式正确。")
                    self.det_model_combo.setCurrentIndex(0)
                    self.model = None
                finally:
                    QApplication.restoreOverrideCursor()
                return
            else:
                # 如果用户取消选择，保持当前选项
                return
        
        # 获取模型路径
        idx = self.det_model_combo.currentIndex()
        full_path = self.det_model_combo.itemData(idx)
        self.model_path = full_path if full_path else text

        # 加载模型 (显示忙碌光标)
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            print(f"Loading model: {self.model_path}")
            self.model = YOLO(self.model_path)
            # 如果线程正在运行，实时更新线程中的模型
            if self.video_thread and self.video_thread.isRunning():
                self.video_thread.set_model(self.model)
        except Exception as e:
            QMessageBox.critical(self.parent, "错误", f"模型加载失败: {e}\n请确认文件存在且格式正确。")
            self.det_model_combo.setCurrentIndex(0)
        finally:
            QApplication.restoreOverrideCursor()

    def update_detect_params(self):
        conf = self.conf_slider.value() / 100.0
        iou = self.iou_slider.value() / 100.0
        self.conf_label.setText(f"置信度: {conf:.2f}")
        self.iou_label.setText(f"IoU 阈值: {iou:.2f}")
        
        # 实时更新运行中线程的参数
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.set_params(conf, iou)


    def update_table_data(self, detections):
        self.result_table.setRowCount(0)
        self.result_table.setRowCount(len(detections))
        for i, det in enumerate(detections):
            item_cls = QTableWidgetItem(det['class'])
            item_cls.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.result_table.setItem(i, 0, item_cls)

            conf_val = f"{det['conf']:.2%}"
            item_conf = QTableWidgetItem(conf_val)
            item_conf.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_conf.setForeground(QColor("#4caf50") if det['conf'] > 0.7 else QColor("#ff9800"))
            self.result_table.setItem(i, 1, item_conf)

            box = det['box']
            coord_str = f"({int(box[0])}, {int(box[1])})"
            item_coord = QTableWidgetItem(coord_str)
            item_coord.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.result_table.setItem(i, 2, item_coord)

    def start_detection(self):
        """根据选择的模式开始识别"""
        mode = self.detect_mode_combo.currentIndex()
        
        if mode == 0:  # 图片识别
            self.process_image_file()
        elif mode == 1:  # 摄像头识别
            self.start_camera()
        elif mode == 2:  # 桌面识别
            self.start_screen()
    
    def open_image(self):
        self.stop_detection()
        # 修改文件选择对话框，支持图片和MP4视频文件
        fname, _ = QFileDialog.getOpenFileName(self.parent, '选择图片或视频', '.', "媒体文件 (*.png *.jpg *.jpeg *.mp4)")
        if fname:
            self.current_file = fname
            file_name = Path(fname).name
            
            # 判断文件类型
            import os
            _, ext = os.path.splitext(self.current_file)
            ext = ext.lower()
            
            # 如果是图片文件，直接显示预览
            if ext in ['.png', '.jpg', '.jpeg']:
                img = cv2.imread(self.current_file)
                if img is not None:
                    self.image_label.setPixmap(cv_img_to_qt(img))
                else:
                    self.image_label.setText(f"图片预览失败: {file_name}")
            # 如果是视频文件，显示文本提示
            elif ext == '.mp4':
                self.image_label.setText(f"已选择视频: {file_name}\n点击'开始识别'进行播放")
            else:
                self.image_label.setText(f"已选择文件: {file_name}")

    def process_image_file(self):
        """处理当前选择的图片/视频文件"""
        if not self.current_file:
            QMessageBox.information(self.parent, "提示", "请先选择要识别的文件")
            return
        
        # 判断文件类型
        import os
        _, ext = os.path.splitext(self.current_file)
        ext = ext.lower()
        
        # 如果是视频文件
        if ext == '.mp4':
            self.open_video(self.current_file)
        # 如果是图片文件
        else:
            img = cv2.imread(self.current_file)
            if img is None: 
                self.image_label.setText("图片文件打开失败")
                return
                
            # 确保模型已加载
            if not self.model:
                self.select_detect_model(self.model_path)

            conf = self.conf_slider.value() / 100.0
            iou = self.iou_slider.value() / 100.0
            results = self.model(img, conf=conf, iou=iou)

            annotated_frame = results[0].plot()
            detections = []
            if results[0].boxes:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id] if self.model.names else str(cls_id)
                    detections.append({
                        "class": class_name,
                        "conf": float(box.conf[0]),
                        "box": box.xyxy[0].tolist()
                    })

            self.image_label.setPixmap(cv_img_to_qt(annotated_frame))
            self.update_table_data(detections)
            
            # 更新最新检测结果
            self.latest_frame = annotated_frame
            self.latest_detections = detections
            self.current_file_type = 'image'
            # 启用保存按钮
            self._update_save_button_state()

    def open_video(self, fname):
        """打开并处理视频文件"""
        # 确保模型已加载
        if not self.model:
            self.select_detect_model(self.det_model_combo.currentText())
            if self.model is None:
                return
        
        # 显示处理中状态
        self.image_label.setText("视频处理中...")
        
        # 检查视频文件是否可打开
        cap = cv2.VideoCapture(fname)
        if not cap.isOpened():
            self.image_label.setText("视频文件打开失败")
            return
        cap.release()
        
        # 设置UI为运行状态
        self._set_ui_running(True)
        
        # 创建并启动视频播放线程
        self.video_player_thread = VideoPlayerThread(fname, self.model)
        
        # 设置检测参数
        conf = self.conf_slider.value() / 100.0
        iou = self.iou_slider.value() / 100.0
        self.video_player_thread.set_params(conf, iou)
        
        # 设置播放速度
        speed_value = self.speed_slider.value() / 10.0
        self.video_player_thread.set_speed(speed_value)
        
        # 连接信号
        self.video_player_thread.change_pixmap_signal.connect(self.update_frame)
        self.video_player_thread.playback_finished_signal.connect(self.video_playback_finished)
        
        # 启动线程
        self.video_player_thread.start()
        
        # 显示视频控制按钮
        self.show_video_controls(True)
    
    def video_playback_finished(self):
        """视频播放完成后的处理"""
        # 隐藏视频控制按钮
        self.show_video_controls(False)
        
        # 只有当视频正常播放完毕（而不是手动暂停）时，才显示完成信息
        if self.is_running:
            self.image_label.setText("视频识别完成")
            self._set_ui_running(False)

    def _start_video_thread(self, source_type):
        """启动视频线程的通用方法"""
        self.stop_detection()
        
        # 确保模型已加载
        if self.model is None:
            self.select_detect_model(self.det_model_combo.currentText())
            # 如果自动加载失败（可能是没文件），终止启动
            if self.model is None:
                return 

        self._set_ui_running(True)

        self.video_thread = VideoThread(source_type=source_type)
        # 直接传递已加载的模型对象
        self.video_thread.set_model(self.model)
        
        # 传递当前选中的显示器索引
        self.video_thread.set_monitor(self.monitor_combo.currentIndex())
        
        self.video_thread.set_params(self.conf_slider.value() / 100.0, self.iou_slider.value() / 100.0)
        self.video_thread.change_pixmap_signal.connect(self.update_frame)
        self.video_thread.start()

    def start_camera(self):
        self._start_video_thread('camera')

    def start_screen(self):
        self._start_video_thread('screen')

    def update_frame(self, cv_img, detections):
        if cv_img is None or not cv_img.size: return
        self.image_label.setPixmap(cv_img_to_qt(cv_img))
        self.update_table_data(detections)
        
        # 更新最新检测结果
        self.latest_frame = cv_img
        self.latest_detections = detections
        # 如果是视频模式，设置文件类型
        if hasattr(self, 'video_player_thread') and self.video_player_thread and self.video_player_thread.isRunning():
            self.current_file_type = 'video'
        # 启用保存按钮
        self._update_save_button_state()

    def stop_detection(self):
        # 停止视频播放线程
        if self.video_player_thread and self.video_player_thread.isRunning():
            self.video_player_thread.stop()
            self.video_player_thread.wait()
            self.video_player_thread = None
            # 隐藏视频控制按钮
            self.show_video_controls(False)
        # 停止普通视频线程
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread = None
        
        # 更新UI状态
        self._set_ui_running(False)

    def on_detect_mode_changed(self, index):
        """当识别模式改变时，控制输入源设置的可见性"""
        # 只有选择桌面识别模式（索引2）时才显示输入源设置
        if index == 2:
            self.screen_group.show()
        else:
            self.screen_group.hide()
    
    def _set_ui_running(self, is_running):
        """统一管理按钮状态"""
        self.is_running = is_running
        self.btn_select_file.setEnabled(not is_running)
        self.btn_start.setEnabled(not is_running)
        self.detect_mode_combo.setEnabled(not is_running)
        self.monitor_combo.setEnabled(not is_running) # 运行时锁定屏幕选择
        self.btn_stop.setEnabled(is_running)
        self.det_model_combo.setEnabled(not is_running)
        
    def _update_save_button_state(self):
        """更新保存按钮状态"""
        self.btn_save_result.setEnabled(self.latest_frame is not None)
        
    def toggle_video_pause(self):
        """切换视频播放/暂停状态"""
        if self.video_player_thread:
            self.video_player_thread.toggle_pause()
            # 更新按钮文本
            if self.video_player_thread._pause_flag:
                self.btn_play_pause.setText("▶ 继续")
            else:
                self.btn_play_pause.setText("⏯ 暂停")
    
    def forward_video(self):
        """视频快进5秒"""
        if self.video_player_thread:
            self.video_player_thread.fast_forward(5)
    
    def rewind_video(self):
        """视频后退5秒"""
        if self.video_player_thread:
            self.video_player_thread.rewind(5)
    
    def change_video_speed(self):
        """改变视频播放速度"""
        speed_value = self.speed_slider.value() / 10.0  # 转换为0.1x到3.0x
        self.speed_label.setText(f"{speed_value}x")
        if self.video_player_thread:
            self.video_player_thread.set_speed(speed_value)
    
    def show_video_controls(self, show=True):
        """显示或隐藏视频控制按钮"""
        self.video_controls_group.setVisible(show)
        if show:
            self.btn_play_pause.setText("⏯ 暂停")
        else:
            self.btn_play_pause.setText("⏯ 暂停")
    
    def save_detection_result(self):
        """保存当前检测结果"""
        # 检查latest_frame是否为有效数组
        if self.latest_frame is None or not hasattr(self.latest_frame, 'shape') or self.latest_frame.size == 0:
            QMessageBox.information(self.parent, "提示", "没有可保存的检测结果")
            return
        
        # 根据当前文件类型选择保存方式
        if self.current_file_type == 'image':
            self._save_image_result()
        elif self.current_file_type == 'video':
            self._save_video_result()
        else:
            QMessageBox.information(self.parent, "提示", "当前模式不支持保存结果")
    
    def _save_image_result(self):
        """保存图片检测结果"""
        # 再次检查latest_frame是否有效
        if self.latest_frame is None or not hasattr(self.latest_frame, 'shape') or self.latest_frame.size == 0:
            QMessageBox.information(self.parent, "提示", "没有可保存的检测结果")
            return
        
        # 选择保存路径和格式
        save_path, _ = QFileDialog.getSaveFileName(
            self.parent, 
            "保存检测结果", 
            str(Path.home() / "detection_result.png"), 
            "PNG图片 (*.png);;JPEG图片 (*.jpg);;BMP图片 (*.bmp)"
        )
        
        if not save_path:
            return
        
        # 保存图片
        try:
            # 直接保存BGR格式，因为OpenCV默认使用BGR
            cv2.imwrite(save_path, self.latest_frame)
            
            # 如果有检测结果，保存为JSON文件
            if self.latest_detections:
                import json
                json_path = save_path.rsplit('.', 1)[0] + '.json'
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(self.latest_detections, f, ensure_ascii=False, indent=2)
                
            QMessageBox.information(self.parent, "成功", f"检测结果已保存到:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self.parent, "错误", f"保存图片失败: {str(e)}")
    
    def _save_video_result(self):
        """保存视频检测结果"""
        # 显示保存选项对话框
        save_option = QMessageBox.question(
            self.parent,
            "保存视频结果",
            "请选择保存方式:\n\n" 
            "1. 保存当前帧 (快速)\n" 
            "2. 保存整个视频 (耗时较长)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes
        )
        
        if save_option == QMessageBox.StandardButton.Cancel:
            return
        elif save_option == QMessageBox.StandardButton.Yes:
            # 保存当前帧
            self._save_video_current_frame()
        else:
            # 保存整个视频（TODO：实现完整视频保存）
            QMessageBox.information(self.parent, "提示", "保存整个视频功能正在开发中")
    
    def _save_video_current_frame(self):
        """保存视频当前帧"""
        # 检查latest_frame是否有效
        if self.latest_frame is None or not hasattr(self.latest_frame, 'shape') or self.latest_frame.size == 0:
            QMessageBox.information(self.parent, "提示", "没有可保存的帧")
            return
        
        # 选择保存路径和格式
        save_path, _ = QFileDialog.getSaveFileName(
            self.parent, 
            "保存当前帧", 
            str(Path.home() / "video_frame.png"), 
            "PNG图片 (*.png);;JPEG图片 (*.jpg);;BMP图片 (*.bmp)"
        )
        
        if not save_path:
            return
        
        try:
            # 保存图片
            cv2.imwrite(save_path, self.latest_frame)
            
            # 如果有检测结果，保存为JSON文件
            if self.latest_detections:
                import json
                json_path = save_path.rsplit('.', 1)[0] + '.json'
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(self.latest_detections, f, ensure_ascii=False, indent=2)
            
            QMessageBox.information(self.parent, "成功", f"当前帧已保存到:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self.parent, "错误", f"保存帧失败: {str(e)}")
