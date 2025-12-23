import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (QGroupBox, QFormLayout, QLineEdit, QPushButton, QVBoxLayout,
                             QHBoxLayout, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
                             QAbstractItemView, QLabel, QMessageBox, QWidget, QFileDialog,
                             QSizePolicy)
from PyQt6.QtCore import Qt, QPoint, QRectF
from PyQt6.QtGui import QColor, QImage, QPixmap, QCursor, QPainter, QPen
from .utils import cv_img_to_qt

class AnnotationModule:
    def __init__(self, parent):
        self.parent = parent
        
        # --- 核心状态 ---
        self.drawing = False
        self.boxes = []
        self.current_image_path = None
        self.image_list_data = []
        self.classes = []
        
        # --- 优化1：缓存与未保存检测 ---
        self.current_cv_img = None  # OpenCV 原图缓存
        self.qt_pixmap = None       # Qt Pixmap 缓存 (用于绘图)
        self.is_modified = False    # 是否有未保存的修改
        self.last_selected_row = -1 # 用于取消切换时回滚
        
        # --- 优化3：缩放与平移参数 ---
        self.scale_factor = 1.0     # 当前缩放倍率
        self.offset = QPoint(0, 0)  # 视图偏移量 (x, y)
        self.last_mouse_pos = QPoint() # 上一次鼠标位置 (用于拖拽)
        self.panning = False        # 是否正在平移

        # 绘图过程中的临时变量
        self.start_point = None     # 绘图起点 (图像坐标)
        self.current_box = None     # 正在绘制的框

    def init_ui(self, tab_annotate):
        layout = tab_annotate.layout()
        if not layout:
            layout = QHBoxLayout(tab_annotate)

        # --- 左侧控制面板 ---
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(10)

        # 1. 数据集配置
        dataset_group = QGroupBox("数据集配置")
        dataset_form = QFormLayout()
        self.dataset_dir_edit = QLineEdit()
        self.dataset_dir_edit.setPlaceholderText("选择数据集目录")
        btn_dataset = QPushButton("...")
        btn_dataset.setFixedWidth(40)
        btn_dataset.clicked.connect(self.select_dataset_dir)
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(self.dataset_dir_edit)
        dataset_layout.addWidget(btn_dataset)

        self.classes_file_edit = QLineEdit()
        self.classes_file_edit.setPlaceholderText("选择类别文件(.txt/.yaml)")
        btn_classes = QPushButton("...")
        btn_classes.setFixedWidth(40)
        btn_classes.clicked.connect(self.select_classes_file)
        classes_layout = QHBoxLayout()
        classes_layout.addWidget(self.classes_file_edit)
        classes_layout.addWidget(btn_classes)

        self.btn_load_dataset = QPushButton("📂 加载数据集")
        self.btn_load_dataset.clicked.connect(self.load_dataset)

        dataset_form.addRow("数据集目录:", dataset_layout)
        dataset_form.addRow("类别文件:", classes_layout)
        dataset_form.addRow(self.btn_load_dataset)
        dataset_group.setLayout(dataset_form)

        # 2. 图像列表
        img_list_group = QGroupBox("图像列表")
        img_list_layout = QVBoxLayout()
        self.image_list = QTableWidget()
        self.image_list.setColumnCount(4)
        self.image_list.setHorizontalHeaderLabels(["文件名", "已标注", "数量", "种类"])
        self.image_list.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.image_list.verticalHeader().setVisible(False)
        self.image_list.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.image_list.cellClicked.connect(self.on_image_list_clicked) # 修改事件绑定
        img_list_layout.addWidget(self.image_list)
        img_list_group.setLayout(img_list_layout)

        # 3. 标注工具
        annotate_tools_group = QGroupBox("标注工具")
        annotate_tools_layout = QVBoxLayout()

        self.class_combo = QComboBox()
        self.class_combo.setPlaceholderText("选择类别")
        annotate_tools_layout.addWidget(QLabel("目标类别:"))
        annotate_tools_layout.addWidget(self.class_combo)

        tools_btn_layout = QHBoxLayout()
        self.btn_draw_box = QPushButton("📏 绘制框(L)")
        self.btn_draw_box.setCheckable(True)
        self.btn_draw_box.clicked.connect(self.enable_draw_box)
        self.btn_delete_box = QPushButton("🗑️ 删除框")
        self.btn_delete_box.clicked.connect(self.delete_selected_box)
        self.btn_clear_all = QPushButton("🧹 清空所有")
        self.btn_clear_all.clicked.connect(self.clear_all_boxes)
        tools_btn_layout.addWidget(self.btn_draw_box)
        tools_btn_layout.addWidget(self.btn_delete_box)
        tools_btn_layout.addWidget(self.btn_clear_all)
        annotate_tools_layout.addLayout(tools_btn_layout)

        self.btn_save_annot = QPushButton("💾 保存标注 (Ctrl+S)")
        self.btn_save_annot.setEnabled(False)
        self.btn_save_annot.clicked.connect(self.save_annotation)
        # 绑定快捷键
        self.btn_save_annot.setShortcut("Ctrl+S")
        
        annotate_tools_layout.addWidget(self.btn_save_annot)
        
        # 添加复位视图按钮
        self.btn_reset_view = QPushButton("🔄 复位视图")
        self.btn_reset_view.clicked.connect(self.reset_view_fit)
        annotate_tools_layout.addWidget(self.btn_reset_view)
        
        annotate_tools_group.setLayout(annotate_tools_layout)

        # 4. 标注信息
        annot_info_group = QGroupBox("标注信息")
        annot_info_layout = QVBoxLayout()
        self.annot_info_table = QTableWidget()
        self.annot_info_table.setColumnCount(5)
        self.annot_info_table.setHorizontalHeaderLabels(["类别", "x1", "y1", "x2", "y2"])
        self.annot_info_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.annot_info_table.verticalHeader().setVisible(False)
        self.annot_info_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        annot_info_layout.addWidget(self.annot_info_table)
        annot_info_group.setLayout(annot_info_layout)

        controls_layout.addWidget(dataset_group)
        controls_layout.addWidget(img_list_group)
        controls_layout.addWidget(annotate_tools_group)
        controls_layout.addWidget(annot_info_group)

        # 左侧容器
        left_widget = self.parent.findChild(QWidget, "annotate_left_widget")
        if not left_widget:
            left_widget = QWidget()
            left_widget.setObjectName("annotate_left_widget")
            left_widget.setLayout(controls_layout)
            left_widget.setFixedWidth(350)
            layout.addWidget(left_widget)
        else:
             # 如果重复初始化，清空旧布局重建
            left_layout = left_widget.layout()
            while left_layout.count() > 0:
                item = left_layout.takeAt(0)
                if item.widget(): item.widget().deleteLater()
            left_layout.addWidget(dataset_group)
            left_layout.addWidget(img_list_group)
            left_layout.addWidget(annotate_tools_group)
            left_layout.addWidget(annot_info_group)

        # --- 右侧标注区域 ---
        annotate_layout = self.parent.findChild(QVBoxLayout, "annotate_display_layout")
        if not annotate_layout:
            annotate_layout = QVBoxLayout()
            annotate_layout.setObjectName("annotate_display_layout")

            # 图像显示 Label
            self.annotate_image_label = QLabel("请加载数据集\n(滚轮缩放，右键拖拽)")
            self.annotate_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.annotate_image_label.setStyleSheet("background-color: #2b2b2b; border: 2px solid #555; border-radius: 4px;")
            
            # 开启鼠标追踪
            self.annotate_image_label.setMouseTracking(True)
            
            # 绑定事件 (使用 Monkey Patching 方式覆盖 Label 的事件)
            self.annotate_image_label.mousePressEvent = self.on_mouse_press
            self.annotate_image_label.mouseMoveEvent = self.on_mouse_move
            self.annotate_image_label.mouseReleaseEvent = self.on_mouse_release
            self.annotate_image_label.wheelEvent = self.on_wheel_event
            # 需要在 Label 上重绘时触发 (例如 Resize)
            self.annotate_image_label.paintEvent = self.on_paint_event 

            # 设置图像标签的尺寸策略
            self.annotate_image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            annotate_layout.addWidget(self.annotate_image_label)

            # 状态栏
            status_layout = QHBoxLayout()
            self.status_label = QLabel("就绪")
            self.status_label.setStyleSheet("color: #00bcd4; font-weight: bold; border: 1px solid #444; border-radius: 3px; padding: 2px 8px; background-color: rgba(50, 50, 50, 0.8);")
            self.status_label.setContentsMargins(0, 5, 0, 5)  # 减少上下边距
            status_layout.addWidget(self.status_label)
            status_layout.addStretch()
            status_layout.setContentsMargins(0, 0, 0, 0)  # 移除布局边距
            
            # 将状态栏添加到布局并设置靠下对齐
            annotate_layout.addLayout(status_layout)
            annotate_layout.setAlignment(status_layout, Qt.AlignmentFlag.AlignBottom)
            
            # 为整个标注区域添加边框
            annotate_layout.setContentsMargins(5, 5, 5, 5)
            annotate_widget = QWidget()
            annotate_widget.setObjectName("annotate_widget")
            annotate_widget.setLayout(annotate_layout)
            annotate_widget.setStyleSheet("border: 2px solid #666; border-radius: 6px; background-color: #222;")
            layout.addWidget(annotate_widget)
        else:
            # 查找并重置现有组件
            # 如果annotate_layout是嵌套在annotate_widget中的情况
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if isinstance(item.widget(), QWidget) and item.widget().objectName() == "annotate_widget":
                    annotate_widget = item.widget()
                    inner_annotate_layout = annotate_widget.layout()
                    for j in range(inner_annotate_layout.count()):
                        inner_item = inner_annotate_layout.itemAt(j)
                        if inner_item.widget() and isinstance(inner_item.widget(), QLabel):
                            if not getattr(self, 'annotate_image_label', None):
                                self.annotate_image_label = inner_item.widget()
                            elif not getattr(self, 'status_label', None):
                                self.status_label = inner_item.widget()
                    break
            
            # 重新应用事件绑定
            if hasattr(self, 'annotate_image_label'):
                self.annotate_image_label.setMouseTracking(True)
                self.annotate_image_label.mousePressEvent = self.on_mouse_press
                self.annotate_image_label.mouseMoveEvent = self.on_mouse_move
                self.annotate_image_label.mouseReleaseEvent = self.on_mouse_release
                self.annotate_image_label.wheelEvent = self.on_wheel_event
                self.annotate_image_label.paintEvent = self.on_paint_event

    # ================= 辅助逻辑 =================


    def select_dataset_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self.parent, "选择数据集目录")
        if dir_path:
            self.dataset_dir_edit.setText(dir_path)
            # 自动寻找 yaml
            import yaml
            dataset_path = Path(dir_path)
            yaml_files = list(dataset_path.rglob("*.yaml")) + list(dataset_path.rglob("*.yml"))
            for yf in yaml_files:
                try:
                    with open(yf, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                        if 'names' in data:
                            self.classes_file_edit.setText(str(yf))
                            self.load_classes(str(yf))
                            break
                except: continue

    def select_classes_file(self):
        fname, _ = QFileDialog.getOpenFileName(self.parent, "选择类别文件", ".", "配置文件 (*.txt *.yaml *.yml)")
        if fname:
            self.classes_file_edit.setText(fname)
            self.load_classes(fname)

    def load_classes(self, file_path):
        try:
            file_ext = Path(file_path).suffix.lower()
            if file_ext in ['.yaml', '.yml']:
                import yaml
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if 'names' in data:
                    names = data['names']
                    if isinstance(names, list):
                        self.classes = [str(n) for n in names]
                    elif isinstance(names, dict):
                        self.classes = [str(names[k]) for k in sorted(names.keys())]
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.classes = [line.strip() for line in f if line.strip()]
            
            self.class_combo.clear()
            self.class_combo.addItems(self.classes)
            self.status_label.setText(f"已加载 {len(self.classes)} 个类别")
        except Exception as e:
            QMessageBox.critical(self.parent, "错误", f"加载类别失败: {e}")

    def load_dataset(self):
        dataset_dir = self.dataset_dir_edit.text()
        if not dataset_dir: return

        path = Path(dataset_dir)
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        img_files = sorted([p for p in path.rglob("*") if p.suffix.lower() in img_exts])
        
        self.image_list_data = []
        for img_path in img_files:
            txt_path = None
            # 查找逻辑
            if img_path.with_suffix('.txt').exists():
                txt_path = img_path.with_suffix('.txt')
            else:
                try:
                    if 'images' in img_path.parts:
                        parts = list(img_path.parts)
                        idx = len(parts) - 1 - parts[::-1].index('images')
                        parts[idx] = 'labels'
                        possible = Path(*parts).with_suffix('.txt')
                        if possible.exists():
                            txt_path = possible
                except: pass

            annot_count = 0
            has_annot = False
            label_types = set()
            
            if txt_path:
                has_annot = True
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        lines = [l.strip() for l in f if l.strip()]
                        annot_count = len(lines)
                        for line in lines:
                            cid = int(line.split()[0])
                            if cid < len(self.classes):
                                label_types.add(self.classes[cid])
                            else:
                                label_types.add(str(cid))
                except: pass
            
            self.image_list_data.append({
                'path': str(img_path),
                'name': img_path.name,
                'has_annotation': has_annot,
                'annot_count': annot_count,
                'label_types': list(label_types),
                'txt_path': str(txt_path) if txt_path else None
            })

        self.update_image_list_ui()
        self.last_selected_row = -1
        self.current_image_path = None
        self.boxes = []
        self.is_modified = False
        self.qt_pixmap = None
        self.annotate_image_label.update() # 触发重绘
        self.status_label.setText(f"已加载 {len(self.image_list_data)} 张图像")

    def update_image_list_ui(self):
        self.image_list.setRowCount(len(self.image_list_data))
        for i, data in enumerate(self.image_list_data):
            self.image_list.setItem(i, 0, QTableWidgetItem(data['name']))
            item_status = QTableWidgetItem("是" if data['has_annotation'] else "否")
            item_status.setForeground(QColor("#4caf50") if data['has_annotation'] else QColor("#ff9800"))
            item_status.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_list.setItem(i, 1, item_status)
            self.image_list.setItem(i, 2, QTableWidgetItem(str(data['annot_count'])))
            self.image_list.setItem(i, 3, QTableWidgetItem(",".join(data['label_types'])))

    # ================= 优化2：切换图片与未保存检测 =================

    def check_unsaved_changes(self):
        """检查是否有未保存的修改"""
        if self.is_modified:
            reply = QMessageBox.question(
                self.parent, "未保存的更改", 
                "当前图片有未保存的标注，是否保存？\n(选择'否'将丢弃更改，'取消'将留在当前图片)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.save_annotation()
                return True
            elif reply == QMessageBox.StandardButton.No:
                return True # 允许切换，不保存
            else:
                return False # 取消切换
        return True

    def on_image_list_clicked(self, row, column):
        # 如果点击的是当前行，不做处理
        if row == self.last_selected_row:
            return

        # 检查未保存
        if not self.check_unsaved_changes():
            # 恢复选中状态到上一行
            if self.last_selected_row != -1:
                self.image_list.selectRow(self.last_selected_row)
            else:
                self.image_list.clearSelection()
            return

        self.load_image_data(row)

    def load_image_data(self, row):
        if row < 0: return
        self.last_selected_row = row
        data = self.image_list_data[row]
        self.current_image_path = data['path']
        
        # 优化1：只读取一次硬盘
        self.current_cv_img = cv2.imread(self.current_image_path)
        if self.current_cv_img is None: 
            QMessageBox.warning(self.parent, "错误", "无法读取图像")
            return
            
        self.qt_pixmap = cv_img_to_qt(self.current_cv_img)
        self.img_height, self.img_width = self.current_cv_img.shape[:2]
        
        self.boxes = []
        self.current_box = None
        self.is_modified = False
        
        # 解析标注
        if data['txt_path'] and Path(data['txt_path']).exists():
            try:
                with open(data['txt_path'], 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            xc, yc, w, h = map(float, parts[1:5])
                            
                            x1 = int((xc - w/2) * self.img_width)
                            y1 = int((yc - h/2) * self.img_height)
                            x2 = int((xc + w/2) * self.img_width)
                            y2 = int((yc + h/2) * self.img_height)
                            
                            cname = self.classes[cid] if cid < len(self.classes) else str(cid)
                            self.boxes.append({
                                'class_id': cid, 'class_name': cname,
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                            })
            except Exception as e:
                print(f"标注解析失败: {e}")

        self.update_annot_info_table()
        self.btn_save_annot.setEnabled(True)
        self.status_label.setText(f"正在标注: {data['name']}")
        
        # 初始视图复位
        self.reset_view_fit()

    # ================= 优化3：视图变换与渲染 (QPainter) =================

    def reset_view_fit(self):
        """重置视图以适应窗口"""
        if self.qt_pixmap is None: return
        
        label_w = self.annotate_image_label.width()
        label_h = self.annotate_image_label.height()
        if label_w == 0 or label_h == 0: return # 防止除零

        scale_w = label_w / self.img_width
        scale_h = label_h / self.img_height
        self.scale_factor = min(scale_w, scale_h) * 0.95 # 留一点边距
        
        # 居中偏移
        new_w = self.img_width * self.scale_factor
        new_h = self.img_height * self.scale_factor
        self.offset = QPoint(int((label_w - new_w) / 2), int((label_h - new_h) / 2))
        
        self.annotate_image_label.update() # 触发 PaintEvent

    def img_to_screen(self, x, y):
        """图像坐标 -> 屏幕坐标"""
        sx = x * self.scale_factor + self.offset.x()
        sy = y * self.scale_factor + self.offset.y()
        return int(sx), int(sy)

    def screen_to_img(self, sx, sy):
        """屏幕坐标 -> 图像坐标"""
        ix = (sx - self.offset.x()) / self.scale_factor
        iy = (sy - self.offset.y()) / self.scale_factor
        return int(ix), int(iy)

    def on_paint_event(self, event):
        """核心渲染函数：替代原来的 redraw_image"""
        painter = QPainter(self.annotate_image_label)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False) # 提高性能，像素风
        
        # 1. 绘制背景
        painter.fillRect(self.annotate_image_label.rect(), QColor("#2b2b2b"))
        
        if self.qt_pixmap is None:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.annotate_image_label.rect(), Qt.AlignmentFlag.AlignCenter, "请选择图像")
            return

        # 2. 绘制图像 (应用缩放和平移)
        # 目标矩形 (屏幕上的位置)
        target_rect = QRectF(
            self.offset.x(), self.offset.y(),
            self.img_width * self.scale_factor,
            self.img_height * self.scale_factor
        )
        # 源矩形 (整张图)
        source_rect = QRectF(0, 0, self.img_width, self.img_height)
        
        painter.drawPixmap(target_rect, self.qt_pixmap, source_rect)

        # 3. 绘制已有的框
        pen_box = QPen(QColor(0, 255, 0), 2)
        painter.setPen(pen_box)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)

        for box in self.boxes:
            # 转换坐标
            x1, y1 = self.img_to_screen(box['x1'], box['y1'])
            x2, y2 = self.img_to_screen(box['x2'], box['y2'])
            w, h = x2 - x1, y2 - y1
            
            painter.drawRect(x1, y1, w, h)
            # 绘制标签背景
            label_text = box['class_name']
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(label_text)
            th = fm.height()
            painter.fillRect(x1, y1 - th, tw + 4, th, QColor(0, 255, 0))
            
            painter.save()
            painter.setPen(Qt.GlobalColor.black)
            painter.drawText(x1 + 2, y1 - 2, label_text)
            painter.restore()

        # 4. 绘制当前正在画的框
        if self.current_box:
            pen_curr = QPen(QColor(0, 0, 255), 2)
            # 虚线效果
            pen_curr.setStyle(Qt.PenStyle.DashLine) 
            painter.setPen(pen_curr)
            
            x1, y1 = self.img_to_screen(self.current_box['x1'], self.current_box['y1'])
            x2, y2 = self.img_to_screen(self.current_box['x2'], self.current_box['y2'])
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

    # ================= 交互事件处理 =================

    def on_wheel_event(self, event):
        """滚轮缩放"""
        if self.qt_pixmap is None: return

        angle = event.angleDelta().y()
        zoom_in = angle > 0
        
        old_scale = self.scale_factor
        zoom_rate = 1.1 if zoom_in else 0.9
        self.scale_factor *= zoom_rate
        
        # 限制缩放范围
        self.scale_factor = max(0.01, min(self.scale_factor, 50.0))
        
        # 以鼠标为中心缩放
        mouse_pos = event.position()
        
        # 原理: (mouse - offset_old) / scale_old = img_point = (mouse - offset_new) / scale_new
        # offset_new = mouse - img_point * scale_new
        
        vec_x = mouse_pos.x() - self.offset.x()
        vec_y = mouse_pos.y() - self.offset.y()
        
        self.offset.setX(int(mouse_pos.x() - vec_x * (self.scale_factor / old_scale)))
        self.offset.setY(int(mouse_pos.y() - vec_y * (self.scale_factor / old_scale)))
        
        self.annotate_image_label.update()

    def on_mouse_press(self, event):
        if self.qt_pixmap is None: return
        
        # 右键或中键拖拽
        if event.button() == Qt.MouseButton.RightButton or event.button() == Qt.MouseButton.MiddleButton:
            self.panning = True
            self.last_mouse_pos = event.position().toPoint()
            self.annotate_image_label.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        # 左键处理
        if event.button() == Qt.MouseButton.LeftButton:
            # 如果是绘制模式
            if self.drawing:
                ix, iy = self.screen_to_img(event.position().x(), event.position().y())
                # 限制在图片范围内
                ix = max(0, min(ix, self.img_width))
                iy = max(0, min(iy, self.img_height))
                
                self.start_point = (ix, iy)
                self.current_box = {'x1': ix, 'y1': iy, 'x2': ix, 'y2': iy}
                self.annotate_image_label.update()
            else:
                # 非绘制模式下左键也可以是拖拽，或者仅仅是选中框(暂未实现选中单个框)
                self.panning = True
                self.last_mouse_pos = event.position().toPoint()
                self.annotate_image_label.setCursor(Qt.CursorShape.ClosedHandCursor)

    def on_mouse_move(self, event):
        if self.qt_pixmap is None: return

        if self.panning:
            delta = event.position().toPoint() - self.last_mouse_pos
            self.offset += delta
            self.last_mouse_pos = event.position().toPoint()
            self.annotate_image_label.update()
            return

        if self.drawing and self.start_point:
            ix, iy = self.screen_to_img(event.position().x(), event.position().y())
            ix = max(0, min(ix, self.img_width))
            iy = max(0, min(iy, self.img_height))
            
            self.current_box = {
                'x1': min(self.start_point[0], ix),
                'y1': min(self.start_point[1], iy),
                'x2': max(self.start_point[0], ix),
                'y2': max(self.start_point[1], iy)
            }
            self.annotate_image_label.update()

    def on_mouse_release(self, event):
        if self.qt_pixmap is None: return

        if self.panning:
            self.panning = False
            cursor = Qt.CursorShape.CrossCursor if self.drawing else Qt.CursorShape.ArrowCursor
            self.annotate_image_label.setCursor(cursor)
            return

        if self.drawing and self.start_point:
            if self.class_combo.currentIndex() == -1:
                QMessageBox.warning(self.parent, "提示", "请先选择类别")
                self.cancel_drawing()
                self.annotate_image_label.update()
                return

            ix, iy = self.screen_to_img(event.position().x(), event.position().y())
            ix = max(0, min(ix, self.img_width))
            iy = max(0, min(iy, self.img_height))
            
            x1 = min(self.start_point[0], ix)
            y1 = min(self.start_point[1], iy)
            x2 = max(self.start_point[0], ix)
            y2 = max(self.start_point[1], iy)

            if x2 - x1 > 2 and y2 - y1 > 2: # 忽略极小框
                cid = self.class_combo.currentIndex()
                self.boxes.append({
                    'class_id': cid,
                    'class_name': self.class_combo.currentText(),
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })
                self.is_modified = True # 标记已修改
                self.update_annot_info_table()
                # 更新列表显示数量
                if self.last_selected_row >= 0:
                    self.image_list_data[self.last_selected_row]['annot_count'] = len(self.boxes)
                    self.update_image_list_ui_item(self.last_selected_row)
            
            self.cancel_drawing()
            self.annotate_image_label.update()

    def cancel_drawing(self):
        self.drawing = False
        self.btn_draw_box.setChecked(False)
        self.annotate_image_label.setCursor(Qt.CursorShape.ArrowCursor)
        self.start_point = None
        self.current_box = None
        self.status_label.setText("模式: 浏览 (右键拖拽，滚轮缩放)")

    # ================= 其他功能 =================

    def update_annot_info_table(self):
        self.annot_info_table.setRowCount(len(self.boxes))
        for i, box in enumerate(self.boxes):
            self.annot_info_table.setItem(i, 0, QTableWidgetItem(box['class_name']))
            self.annot_info_table.setItem(i, 1, QTableWidgetItem(str(box['x1'])))
            self.annot_info_table.setItem(i, 2, QTableWidgetItem(str(box['y1'])))
            self.annot_info_table.setItem(i, 3, QTableWidgetItem(str(box['x2'])))
            self.annot_info_table.setItem(i, 4, QTableWidgetItem(str(box['y2'])))

    def update_image_list_ui_item(self, row):
        """仅更新列表中的单行，避免全量刷新"""
        data = self.image_list_data[row]
        item_status = QTableWidgetItem("是" if len(self.boxes) > 0 else "否")
        item_status.setForeground(QColor("#4caf50") if len(self.boxes) > 0 else QColor("#ff9800"))
        item_status.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_list.setItem(row, 1, item_status)
        self.image_list.setItem(row, 2, QTableWidgetItem(str(len(self.boxes))))

    def enable_draw_box(self, checked):
        if not self.qt_pixmap:
            self.btn_draw_box.setChecked(False)
            return
        self.drawing = checked
        if checked:
            self.annotate_image_label.setCursor(Qt.CursorShape.CrossCursor)
            self.status_label.setText("模式: 绘制中 (按住左键拖动)")
        else:
            self.annotate_image_label.setCursor(Qt.CursorShape.ArrowCursor)
            self.status_label.setText("模式: 浏览")

    def delete_selected_box(self):
        row = self.annot_info_table.currentRow()
        if row >= 0 and row < len(self.boxes):
            del self.boxes[row]
            self.is_modified = True
            self.update_annot_info_table()
            self.annotate_image_label.update()
            if self.last_selected_row >= 0:
                self.image_list_data[self.last_selected_row]['annot_count'] = len(self.boxes)
                self.update_image_list_ui_item(self.last_selected_row)

    def clear_all_boxes(self):
        if not self.boxes: return
        if QMessageBox.question(self.parent, "确认", "确定清空当前图片所有标注？") == QMessageBox.StandardButton.Yes:
            self.boxes = []
            self.is_modified = True
            self.update_annot_info_table()
            self.annotate_image_label.update()
            if self.last_selected_row >= 0:
                self.image_list_data[self.last_selected_row]['annot_count'] = 0
                self.image_list_data[self.last_selected_row]['has_annotation'] = False
                self.update_image_list_ui_item(self.last_selected_row)

    def save_annotation(self):
        if not self.current_image_path: return
        
        img_path = Path(self.current_image_path)
        save_path = img_path.with_suffix('.txt')
        
        # 路径逻辑同前
        curr_row = self.last_selected_row
        if curr_row >= 0 and self.image_list_data[curr_row]['txt_path']:
             save_path = Path(self.image_list_data[curr_row]['txt_path'])
        else:
            if 'images' in img_path.parts:
                try:
                    parts = list(img_path.parts)
                    idx = len(parts) - 1 - parts[::-1].index('images')
                    parts[idx] = 'labels'
                    label_path = Path(*parts).with_suffix('.txt')
                    if label_path.parent.exists():
                        save_path = label_path
                except: pass
            
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                for box in self.boxes:
                    xc = (box['x1'] + box['x2']) / 2.0 / self.img_width
                    yc = (box['y1'] + box['y2']) / 2.0 / self.img_height
                    w = (box['x2'] - box['x1']) / float(self.img_width)
                    h = (box['y2'] - box['y1']) / float(self.img_height)
                    f.write(f"{box['class_id']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            
            self.status_label.setText(f"已保存: {save_path.name}")
            self.is_modified = False # 重置修改标记
            
            if curr_row >= 0:
                self.image_list_data[curr_row]['txt_path'] = str(save_path)
                self.image_list_data[curr_row]['has_annotation'] = len(self.boxes) > 0
                self.update_image_list_ui_item(curr_row)
                
        except Exception as e:
            QMessageBox.critical(self.parent, "错误", f"保存失败: {e}")