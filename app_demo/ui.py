import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QMessageBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
import time
from app_demo.mode import AgentSim  # Make sure this import works correctly

class LoopThread(QThread):
    image_update = pyqtSignal(np.ndarray)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.cmd = 0

    def run(self):
        i = 0
        while self.running:
            i += 1
            print(f"time step {i} cmd: {self.cmd}")
            image_array = self.get_image_array()
            self.image_update.emit(image_array)
            time.sleep(0.2)
        print("stopped.")

    def get_image_array(self):
        width, height = 1024, 512
        image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        return image_array

class CarControlUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Car Control UI")
        self.setGeometry(100, 100, 1024, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Folder selection
        self.folder_combo = QComboBox()
        layout.addWidget(QLabel("Select Folder:"))
        layout.addWidget(self.folder_combo)

        # Zip file selection
        self.zip_combo = QComboBox()
        layout.addWidget(QLabel("Select Zip File:"))
        layout.addWidget(self.zip_combo)

        # Load Agent button
        self.load_agent_button = QPushButton("Load Agent")
        layout.addWidget(self.load_agent_button)

        self.cmd_label = QLabel("Current cmd: 0")
        layout.addWidget(self.cmd_label)

        button_layout = QHBoxLayout()
        self.button_left = QPushButton("left")
        self.button_forward = QPushButton("forward")
        self.button_right = QPushButton("right")
        button_layout.addWidget(self.button_left)
        button_layout.addWidget(self.button_forward)
        button_layout.addWidget(self.button_right)
        layout.addLayout(button_layout)

        self.start_button = QPushButton("Start Loop")
        self.stop_button = QPushButton("Stop Loop")
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        self.button_left.clicked.connect(lambda: self.set_cmd(1))
        self.button_forward.clicked.connect(lambda: self.set_cmd(0))
        self.button_right.clicked.connect(lambda: self.set_cmd(2))
        self.start_button.clicked.connect(self.start_loop)
        self.stop_button.clicked.connect(self.stop_loop)
        self.folder_combo.currentTextChanged.connect(self.update_zip_files)
        self.load_agent_button.clicked.connect(self.load_agent)

        self.loop_thread = LoopThread()
        self.loop_thread.image_update.connect(self.display_image)

        self.populate_folder_combo()
        self.agent = None

    def populate_folder_combo(self):
        rlmodel_path = "RLmodel"  # Adjust this path if needed
        if os.path.exists(rlmodel_path) and os.path.isdir(rlmodel_path):
            folders = [f for f in os.listdir(rlmodel_path) if os.path.isdir(os.path.join(rlmodel_path, f))]
            self.folder_combo.addItems(folders)

    def update_zip_files(self, folder):
        self.zip_combo.clear()
        folder_path = os.path.join("RLmodel", folder)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            zip_files = [f for f in os.listdir(folder_path) if f.endswith('.zip')]
            self.zip_combo.addItems(zip_files)

    def load_agent(self):
        pass
        # selected_folder = self.folder_combo.currentText()
        # selected_zip = self.zip_combo.currentText()
        
        # if not selected_folder or not selected_zip:
        #     QMessageBox.warning(self, "Warning", "Please select both a folder and a zip file.")
        #     return

        # zip_path = os.path.join("RLmodel", selected_folder, selected_zip)
        
        # try:
        #     self.agent = AgentSim(zip_path)
        #     QMessageBox.information(self, "Success", f"Agent loaded successfully from {zip_path}")
        # except Exception as e:
        #     QMessageBox.critical(self, "Error", f"Failed to load agent: {str(e)}")

    def set_cmd(self, value):
        self.loop_thread.cmd = value
        self.cmd_label.setText(f"Current cmd: {value}")

    def start_loop(self):
        if not self.loop_thread.running:
            self.loop_thread.running = True
            self.loop_thread.start()

    def stop_loop(self):
        self.loop_thread.running = False
        self.loop_thread.wait()

    def display_image(self, image_array):
        height, width, channel = image_array.shape
        bytes_per_line = 3 * width
        q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CarControlUI()
    window.show()
    sys.exit(app.exec_())