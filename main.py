import sys
import time
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QIcon, QImage, QPixmap, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QStackedWidget, 
    QFrame, QSlider, QSystemTrayIcon, QMenu, QStyle
)

# Optional OpenCV import (agar camera installed hai toh video stream dikhane ke liye)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# -------------------------------------------------------------
# 1. Camera Live-Stream Thread (Framework for Modules)
# -------------------------------------------------------------
class CameraThread(QThread):
    # Signals to update UI without freezing Main Thread
    frame_signal = Signal(QImage)
    stats_signal = Signal(float, float) # FPS, Latency (ms)

    def __init__(self):
        super().__init__()
        self.is_running = False

    def run(self):
        if not CV2_AVAILABLE:
            return

        cap = cv2.VideoCapture(0)
        self.is_running = True

        prev_time = time.time()

        while self.is_running and cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # Frame preprocessing (Flip horizontally)
            frame = cv2.flip(frame, 1)

            # --- MODULE HOOK PLACEHOLDER ---
            # Ayush ya dusre members yahan apna gesture detection code add kar sakte hain
            # -------------------------------

            # Convert BGR to RGB for PySide6 QImage
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Performance Metrics Calculation
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # ms
            fps = 1.0 / (end_time - prev_time) if (end_time - prev_time) > 0 else 30.0
            prev_time = end_time

            # Emit Signals to Main UI
            self.frame_signal.emit(q_img)
            self.stats_signal.emit(fps, latency)

            time.sleep(0.01) # Smooth CPU load

        cap.release()

    def stop(self):
        self.is_running = False
        self.wait()


# -------------------------------------------------------------
# 2. Main FlowSync Application Window
# -------------------------------------------------------------
class FlowSyncApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FlowSync • Master Suite")
        self.setGeometry(100, 100, 1050, 680)

        # Inline Dark Theme Styling
        self.setStyleSheet("""
            QMainWindow { background-color: #0E1017; }
            QWidget { color: #C5C6D0; font-family: 'Segoe UI', sans-serif; }
            QFrame#Sidebar { background-color: #12141D; border-right: 1px solid #1D212F; }
            QPushButton#NavBtn {
                background-color: transparent; border: none; padding: 12px;
                border-radius: 8px; font-size: 16px; color: #6C7289;
            }
            QPushButton#NavBtn:hover { background-color: #1D212F; color: #FFFFFF; }
            QPushButton#NavBtn:checked { background-color: #1D212F; color: #5865F2; }
            QFrame#Card {
                background-color: #151822; border: 1px solid #1E2333;
                border-radius: 12px; padding: 16px;
            }
            QPushButton#PrimaryBtn {
                background-color: #5865F2; color: #FFFFFF; font-weight: bold;
                border-radius: 8px; padding: 10px 16px; border: none;
            }
            QPushButton#PrimaryBtn:hover { background-color: #4752C4; }
            QPushButton#DangerBtn {
                background-color: #ED4245; color: #FFFFFF; font-weight: bold;
                border-radius: 8px; padding: 10px 16px; border: none;
            }
            QPushButton#DangerBtn:hover { background-color: #C03537; }
            QLabel#Title { font-size: 20px; font-weight: bold; color: #FFFFFF; }
            QLabel#SubTitle { font-size: 13px; color: #6C7289; }
            QLabel#StatVal { font-size: 16px; font-weight: bold; color: #00E676; }
        """)

        # Main Root Layout
        root_widget = QWidget()
        self.setCentralWidget(root_widget)
        root_layout = QHBoxLayout(root_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # -------------------------------------------------------------
        # Sidebar Navigation Panel
        # -------------------------------------------------------------
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(64)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 16, 8, 16)

        self.btn_dash = QPushButton("🏠")
        self.btn_hand = QPushButton("🖐️")
        self.btn_eye = QPushButton("👁️")
        self.btn_ai = QPushButton("⚡")
        self.btn_settings = QPushButton("⚙️")

        self.nav_buttons = [self.btn_dash, self.btn_hand, self.btn_eye, self.btn_ai, self.btn_settings]
        for btn in self.nav_buttons:
            btn.setObjectName("NavBtn")
            btn.setCheckable(True)
            sidebar_layout.addWidget(btn)

        self.btn_dash.setChecked(True)
        sidebar_layout.addStretch()
        root_layout.addWidget(sidebar)

        # -------------------------------------------------------------
        # Central Stacked Pages
        # -------------------------------------------------------------
        self.stack = QStackedWidget()
        root_layout.addWidget(self.stack)

        self.init_dashboard_page()
        self.init_hand_page()
        self.init_eye_page()
        self.init_ai_page()
        self.init_settings_page()

        # Connect Navigation Click Events
        self.btn_dash.clicked.connect(lambda: self.switch_page(0, self.btn_dash))
        self.btn_hand.clicked.connect(lambda: self.switch_page(1, self.btn_hand))
        self.btn_eye.clicked.connect(lambda: self.switch_page(2, self.btn_eye))
        self.btn_ai.clicked.connect(lambda: self.switch_page(3, self.btn_ai))
        self.btn_settings.clicked.connect(lambda: self.switch_page(4, self.btn_settings))

        # Camera Thread Setup
        self.camera_thread = CameraThread()
        self.camera_thread.frame_signal.connect(self.update_video_feed)
        self.camera_thread.stats_signal.connect(self.update_stats)

        # Windows System Tray Integration
        self.setup_system_tray()

    # -------------------------------------------------------------
    # Page Switcher Logic
    # -------------------------------------------------------------
    def switch_page(self, index, active_btn):
        for btn in self.nav_buttons:
            btn.setChecked(False)
        active_btn.setChecked(True)
        self.stack.setCurrentIndex(index)

    # -------------------------------------------------------------
    # Page 1: Main Dashboard
    # -------------------------------------------------------------
    def init_dashboard_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(QLabel("FlowSync • Control Center", objectName="Title"))
        layout.addWidget(QLabel("Adaptive Computer Vision & System Orchestrator", objectName="SubTitle"))

        card = QFrame()
        card.setObjectName("Card")
        c_lay = QVBoxLayout(card)
        c_lay.addWidget(QLabel("Developer Profile", objectName="Title"))
        c_lay.addWidget(QLabel("Project Owner: Siddhartha Tiwari"))
        c_lay.addWidget(QLabel("Email: siddharthaa1111@gmail.com"))
        c_lay.addWidget(QLabel("System Status: Main Framework & Application Ready"))
        c_lay.addStretch()
        
        layout.addWidget(card)
        self.stack.addWidget(page)

    # -------------------------------------------------------------
    # Page 2: Hand Control Live Stream & Control Page
    # -------------------------------------------------------------
    def init_hand_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(QLabel("Hand Control Interface", objectName="Title"))

        # Video Box Container
        self.video_container = QFrame()
        self.video_container.setObjectName("Card")
        v_lay = QVBoxLayout(self.video_container)

        self.video_label = QLabel("Camera Feed Offline")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("font-size: 16px; color: #6C7289; min-height: 320px;")
        v_lay.addWidget(self.video_label)

        layout.addWidget(self.video_container)

        # Controls & Metrics Bar
        ctrl_card = QFrame()
        ctrl_card.setObjectName("Card")
        ctrl_lay = QHBoxLayout(ctrl_card)

        self.btn_toggle_cam = QPushButton("Start Camera Tracking", objectName="PrimaryBtn")
        self.btn_toggle_cam.clicked.connect(self.toggle_camera)
        ctrl_lay.addWidget(self.btn_toggle_cam)

        ctrl_lay.addStretch()

        self.lbl_fps = QLabel("FPS: --", objectName="StatVal")
        self.lbl_latency = QLabel("Latency: -- ms", objectName="StatVal")
        ctrl_lay.addWidget(self.lbl_fps)
        ctrl_lay.addSpacing(16)
        ctrl_lay.addWidget(self.lbl_latency)

        layout.addWidget(ctrl_card)
        self.stack.addWidget(page)

    # -------------------------------------------------------------
    # Page 3, 4, 5: Placeholders for Team Members
    # -------------------------------------------------------------
    def init_eye_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.addWidget(QLabel("Eye Tracking Module Slot", objectName="Title"))
        layout.addWidget(QLabel("Pending Eye-Gaze Integration", objectName="SubTitle"))
        self.stack.addWidget(page)

    def init_ai_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.addWidget(QLabel("Sync AI Core Slot", objectName="Title"))
        layout.addWidget(QLabel("Coming Soon...", objectName="SubTitle"))
        self.stack.addWidget(page)

    def init_settings_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.addWidget(QLabel("System Preferences", objectName="Title"))
        self.stack.addWidget(page)

    # -------------------------------------------------------------
    # Camera Control & Signal Slots
    # -------------------------------------------------------------
    def toggle_camera(self):
        if not self.camera_thread.isRunning():
            self.camera_thread.start()
            self.btn_toggle_cam.setText("Stop Camera Tracking")
            self.btn_toggle_cam.setObjectName("DangerBtn")
            self.btn_toggle_cam.setStyle(self.btn_toggle_cam.style())
        else:
            self.camera_thread.stop()
            self.btn_toggle_cam.setText("Start Camera Tracking")
            self.btn_toggle_cam.setObjectName("PrimaryBtn")
            self.btn_toggle_cam.setStyle(self.btn_toggle_cam.style())
            self.video_label.setText("Camera Feed Offline")

    def update_video_feed(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def update_stats(self, fps, latency):
        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        self.lbl_latency.setText(f"Latency: {latency:.1f} ms")

    # -------------------------------------------------------------
    # Windows System Tray & Lifecycle Event Handlers
    # -------------------------------------------------------------
    def setup_system_tray(self):
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))

        tray_menu = QMenu()
        tray_menu.setStyleSheet("QMenu { background-color: #151822; color: #FFF; border: 1px solid #1E2333; } QMenu::item:selected { background-color: #5865F2; }")

        show_action = QAction("Open FlowSync", self)
        show_action.triggered.connect(self.restore_window)

        exit_action = QAction("Exit Application", self)
        exit_action.triggered.connect(QApplication.instance().quit)

        tray_menu.addAction(show_action)
        tray_menu.addSeparator()
        tray_menu.addAction(exit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.on_tray_icon_activated)
        self.tray_icon.show()

    def restore_window(self):
        self.showNormal()
        self.activateWindow()

    def on_tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.Trigger:
            if self.isVisible() and not self.isMinimized():
                self.hide()
            else:
                self.restore_window()

    def changeEvent(self, event):
        if event.type() == event.Type.WindowStateChange:
            if self.isMinimized():
                QTimer.singleShot(0, self.hide)
                self.tray_icon.showMessage(
                    "FlowSync Running in Background",
                    "Application minimized to system tray.",
                    QSystemTrayIcon.Information,
                    1500
                )
        super().changeEvent(event)

    def closeEvent(self, event):
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
        self.tray_icon.hide()
        event.accept()
        QApplication.instance().quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FlowSyncApp()
    window.show()
    sys.exit(app.exec())