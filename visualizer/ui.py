import string

from pathlib import Path

import moderngl_window
import moderngl_window.context.pyqt5.window as qtw

from PyQt5 import QtOpenGL, QtWidgets
from PyQt5.QtCore import QSize, Qt, QTimer, QRect
from PyQt5.QtGui import QScreen, QColor, QFontMetrics
from PyQt5.Qt import QPainter, QWidget, pyqtSlot, QEvent

from config import Config, ConfigVal

from const import WINDOW_SIZE_X, WINDOW_SIZE_Y

_g_ui_widget = None

_g_scaling_factor = 1
def update_scaling_factor(app: QtWidgets.QApplication):
    global _g_scaling_factor

    # Make a test label
    alphabet = string.ascii_lowercase[:14]
    test_label = QtWidgets.QLabel(alphabet)
    test_label.setStyleSheet(app.styleSheet())
    test_label.ensurePolished()

    font_height = test_label.fontMetrics().height()

    _g_scaling_factor = font_height / 13

    print("Scaling factor updated to", _g_scaling_factor)

def get_scaling_factor():
    return _g_scaling_factor

def set_target_size(widget: QtWidgets.QWidget):
    base_size = QSize(*widget.SIZE)
    base_size.setWidth(round(base_size.width() * get_scaling_factor()))
    base_size.setHeight(round(base_size.height() * get_scaling_factor()))
    min_size = widget.sizeHint()

    #if widget.layout() is not None:
    #    min_size = widget.layout().sizeHint()
    #    min_size += QSize(widget.layout().spacing(), widget.layout().spacing()) * 2

    size = QSize(max(base_size.width(), min_size.width()), max(base_size.height(), min_size.height()))
    widget.setFixedSize(size)
    widget.resize(size)

class QConfigVal(QWidget):
    FLOAT_SLIDER_PREC = 100

    def __init__(self, name: str, config_val: ConfigVal):
        QWidget.__init__(self)

        self.name = name
        self.config_val = config_val

        self.setAttribute(Qt.WA_StyledBackground)
        self.setAutoFillBackground(True)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)

        self.label = QtWidgets.QLabel("...")

        self.slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.slider.setFixedHeight(round(10 * get_scaling_factor()))

        self.float_mode = (config_val.max - config_val.min) < 10

        if self.float_mode:
            self.slider.setRange(0, self.FLOAT_SLIDER_PREC)
            val_frac = (config_val.val - config_val.min) / (config_val.max - config_val.min)
            self.slider.setValue(round(val_frac * self.FLOAT_SLIDER_PREC))
        else:
            self.slider.setRange(round(config_val.min), round(config_val.max))
            self.slider.setValue(round(config_val.val))

        self.slider.valueChanged.connect(self.on_val_changed)

        self.on_val_changed()

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.slider)

    def get_beautified_name(self):
        bname = self.name.replace('_''', ' ').capitalize()
        return bname

    @pyqtSlot()
    def on_val_changed(self):
        if self.float_mode:
            slider_frac = self.slider.value() / self.FLOAT_SLIDER_PREC
            self.config_val.val = self.config_val.min + (self.config_val.max - self.config_val.min) * slider_frac
        else:
            self.config_val.val = self.slider.value()
        self.label.setText(self.get_beautified_name() + ": " + str(self.config_val.val))

class QEditConfigWidget(QWidget):
    SIZE = (300, 500)

    def __init__(self, config: Config):
        QWidget.__init__(self)

        self.setAttribute(Qt.WA_StyledBackground)
        self.setAutoFillBackground(True)

        self.setLayout(QtWidgets.QVBoxLayout(self))

        self.text_label = QtWidgets.QLabel("Settings:\n")
        self.layout().addWidget(self.text_label)

        self.config = config

        self.camera_group = QtWidgets.QGroupBox("Camera")
        self.camera_group_layout = QtWidgets.QVBoxLayout(self)
        self.camera_group.setLayout(self.camera_group_layout)
        self.layout().addWidget(self.camera_group)

        for name, obj in self.config.__dict__.items():
            if isinstance(obj, ConfigVal):
                config_val = obj # type: ConfigVal

                widget = QConfigVal(name, config_val)

                if name.startswith("camera_"):
                    self.camera_group_layout.addWidget(widget)

        self.footer_label = QtWidgets.QLabel("\n(Click outside this area to close settings)")
        # TODO: Kinda hacky, ideally use setDisabled(True) and add disabled color to stylesheet?
        self.footer_label.setStyleSheet("color: gray")
        self.layout().addWidget(self.footer_label)

        set_target_size(self)

    def update(self):
        super().update()

class QUIBarWidget(QWidget):
    SIZE = (150, 200) # Increased height for controls

    def __init__(self, parent_window):
        QWidget.__init__(self)

        self.config_edit_popup = None

        self.parent_window = parent_window

        self.setAttribute(Qt.WA_StyledBackground)
        self.setAutoFillBackground(True)

        vbox = QtWidgets.QVBoxLayout() # Changed to VBox for better control

        self.text_label = QtWidgets.QLabel("...")
        vbox.addWidget(self.text_label)

        self.edit_config_button = QtWidgets.QPushButton("Edit Settings")
        self.edit_config_button.clicked.connect(self.on_edit_config)
        vbox.addWidget(self.edit_config_button)
        
        # Car Selector
        self.car_selector_group = QtWidgets.QGroupBox("Target Car")
        car_sel_layout = QtWidgets.QHBoxLayout()
        self.car_selector_group.setLayout(car_sel_layout)
        
        self.car_spinbox = QtWidgets.QSpinBox()
        self.car_spinbox.setRange(0, 5) # Max 6 cars
        self.car_spinbox.setValue(0)
        car_sel_layout.addWidget(QtWidgets.QLabel("ID:"))
        car_sel_layout.addWidget(self.car_spinbox)
        
        vbox.addWidget(self.car_selector_group)
        
        # Controls Group
        self.controls_group = QtWidgets.QGroupBox("Controls")
        controls_layout = QtWidgets.QGridLayout()
        self.controls_group.setLayout(controls_layout)
        
        self.btn_w = QtWidgets.QPushButton("W")
        self.btn_a = QtWidgets.QPushButton("A")
        self.btn_s = QtWidgets.QPushButton("S")
        self.btn_d = QtWidgets.QPushButton("D")
        self.btn_jump = QtWidgets.QPushButton("Jump")
        self.btn_boost = QtWidgets.QPushButton("Boost")
        
        controls_layout.addWidget(self.btn_w, 0, 1)
        controls_layout.addWidget(self.btn_a, 1, 0)
        controls_layout.addWidget(self.btn_s, 1, 1)
        controls_layout.addWidget(self.btn_d, 1, 2)
        controls_layout.addWidget(self.btn_jump, 2, 0, 1, 3)
        controls_layout.addWidget(self.btn_boost, 3, 0, 1, 3)
        
        vbox.addWidget(self.controls_group)

        self.setLayout(vbox)

        set_target_size(self)

        global _g_ui_widget
        _g_ui_widget = self

    def update(self):
        super().update()

    @pyqtSlot()
    def on_edit_config(self):
        self.parent_window.toggle_edit_config()

    def set_text(self, text: str):
        self.text_label.setText(text)

def get_ui() -> QUIBarWidget:
    return _g_ui_widget

class QRSVWindow(QtWidgets.QMainWindow):
    def __init__(self, gl_widget):
        super().__init__()

        self.setWindowTitle("RocketSimVis")

        path = Path(__file__).parent.resolve() / "qt_style_sheet.css"
        self.setStyleSheet(path.read_text())

        # Set the central widget of the Window.
        self.gl_widget = gl_widget
        
        # Create a container widget for the layout
        self.container = QtWidgets.QWidget()
        self.setCentralWidget(self.container)
        
        self.base_layout = QtWidgets.QVBoxLayout(self.container)
        self.base_layout.setContentsMargins(0, 0, 0, 0)
        self.base_layout.setSpacing(0)

        self.bar_widget = QUIBarWidget(self)
        self.base_layout.addWidget(self.bar_widget)
        
        # Add GL widget to layout
        self.base_layout.addWidget(self.gl_widget, 1) # Stretch factor 1 to take available space

        self.edit_config_widget = QEditConfigWidget(self.gl_widget.config)
        # We don't add edit_config_widget to layout if it's meant to be an overlay or popup
        # But original code added it to layout. Let's add it but keep it hidden.
        # Actually, looking at toggle_edit_config, it uses setGeometry, implying absolute positioning?
        # "self.edit_config_widget.setGeometry(0, self.bar_widget.height() + 20, ...)"
        # If it uses setGeometry, it should be a child of the window or container, but NOT in the layout.
        
        self.edit_config_widget.setParent(self.container)
        self.edit_config_widget.hide()

        self.resize(WINDOW_SIZE_X, WINDOW_SIZE_Y)

        self.installEventFilter(self)
        # self.centralWidget().installEventFilter(self) # No longer needed/correct if container is central
        self.gl_widget.installEventFilter(self) # Install on GL widget specifically
        
        self.gl_widget.setFocus()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                press_pos = event.pos()

                # Close config window if we click outside of it
                if self.edit_config_widget.isVisible():
                    if not (press_pos in self.edit_config_widget.geometry()):
                        self.toggle_edit_config()
        elif event.type() == QEvent.KeyPress:
            # Ignore auto-repeat events
            if event.isAutoRepeat():
                return True
            print(f"[DEBUG] KeyPress caught in eventFilter: {event.key()}")
            self.gl_widget.keyPressEvent(event)
            return True
        elif event.type() == QEvent.KeyRelease:
            # Ignore auto-repeat events
            if event.isAutoRepeat():
                return True
            print(f"[DEBUG] KeyRelease caught in eventFilter: {event.key()}")
            self.gl_widget.keyReleaseEvent(event)
            return True

        return super().eventFilter(obj, event)

    def toggle_edit_config(self):
        if not self.edit_config_widget.isVisible():
            self.edit_config_widget.show()

            size = self.edit_config_widget.size()

            # Don't exceed our window size
            size.setWidth(min(size.width(), self.width()))
            size.setHeight(min(size.height(), self.height()))

            self.edit_config_widget.setFixedSize(size)

            self.edit_config_widget.setGeometry(
                0, self.bar_widget.height() + 20,
                size.width(), size.width()
            )
        else:
            self.edit_config_widget.hide()