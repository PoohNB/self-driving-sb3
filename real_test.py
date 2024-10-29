import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QMessageBox, QScrollArea, QGridLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
from app_demo.mode import AgentSim
import traceback

class LoopThread(QThread):
    image_update = pyqtSignal(list)
    
    def __init__(self, agent, parent=None):
        super().__init__(parent)
        self.running = False
        self.cmd = [0]
        self.agent = agent

    def run(self):
        i = 0
        while self.running:
            try:
                i += 1
                print(f"time step {i} cmd: {self.cmd}")
                self.agent.step(self.cmd)
                images = self.agent.get_vision()
                self.image_update.emit(images)
            except Exception as e:
                tb = traceback.format_exc()
  
                print(f"Error in loop thread: {e}")
                print(f"Traceback details:\n{tb}")
                self.running = False
        print("stopped.")

class CarControlUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Car Control UI")
        self.setGeometry(100, 100, 1024, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Folder and Zip selection
        self.folder_combo = QComboBox()
        self.zip_combo = QComboBox()
        layout.addWidget(QLabel("Select Folder:"))
        layout.addWidget(self.folder_combo)
        layout.addWidget(QLabel("Select Zip File:"))
        layout.addWidget(self.zip_combo)

        # Buttons
        self.load_agent_button = QPushButton("Load Agent")
        self.close_agent_button = QPushButton("Close Agent")
        self.start_button = QPushButton("Start Loop")
        self.stop_button = QPushButton("Stop Loop")
        self.start_car_button = QPushButton("Start Car")
        self.stop_car_button = QPushButton("Stop Car")
        layout.addWidget(self.load_agent_button)
        layout.addWidget(self.close_agent_button)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.start_car_button)       
        layout.addWidget(self.stop_car_button)

        self.cmd_label = QLabel("Current cmd: [0]")
        layout.addWidget(self.cmd_label)

        button_layout = QHBoxLayout()
        self.button_left = QPushButton("left")
        self.button_forward = QPushButton("forward")
        self.button_right = QPushButton("right")
        button_layout.addWidget(self.button_left)
        button_layout.addWidget(self.button_forward)
        button_layout.addWidget(self.button_right)
        layout.addLayout(button_layout)

        # Scroll area for images
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QGridLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)

        # Connect signals
        self.button_left.clicked.connect(lambda: self.set_cmd([1]))
        self.button_forward.clicked.connect(lambda: self.set_cmd([0]))
        self.button_right.clicked.connect(lambda: self.set_cmd([2]))
        self.start_button.clicked.connect(self.start_loop)
        self.stop_button.clicked.connect(self.stop_loop)
        self.start_car_button.clicked.connect(self.start_car)
        self.stop_car_button.clicked.connect(self.stop_car)
        self.folder_combo.currentTextChanged.connect(self.update_zip_files)
        self.load_agent_button.clicked.connect(self.load_agent)
        self.close_agent_button.clicked.connect(self.close_agent)

        self.populate_folder_combo()
        self.agent = None
        self.loop_thread = None

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
        if self.agent:
            QMessageBox.warning(self, "Warning", "Please close the current agent before loading a new one.")
            return

        selected_folder = self.folder_combo.currentText()
        selected_zip = self.zip_combo.currentText()
        
        if not selected_folder or not selected_zip:
            QMessageBox.warning(self, "Warning", "Please select both a folder and a zip file.")
            return

        zip_path = os.path.join("RLmodel", selected_folder, selected_zip)
        
        try:
            self.agent = AgentSim(zip_path)
            QMessageBox.information(self, "Success", f"Agent loaded successfully from {zip_path}")
            self.loop_thread = LoopThread(self.agent)
            self.loop_thread.image_update.connect(self.display_images)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load agent: {str(e)}")

    def close_agent(self):
        if self.agent:
            self.stop_loop()
            self.agent.close()
            self.agent = None
            self.loop_thread = None
            QMessageBox.information(self, "Success", "Agent closed successfully")
        else:
            QMessageBox.warning(self, "Warning", "No agent is currently loaded")

    def set_cmd(self, value):
        self.cmd = value
        self.cmd_label.setText(f"Current cmd: {self.cmd}")
        if self.loop_thread:
            self.loop_thread.cmd = value

    def start_loop(self):
        if not self.agent:
            QMessageBox.warning(self, "Warning", "Please load an agent first")
            return

        if not self.loop_thread.running:
            self.agent.reset()
            initial_images = self.agent.get_vision()
            self.display_images(initial_images)
            self.loop_thread.running = True
            self.loop_thread.start()

    def stop_loop(self):
        if self.loop_thread and self.loop_thread.running:
            self.loop_thread.running = False
            self.loop_thread.wait()

    def start_car(self):
        self.agent.start()

    def stop_car(self):
        self.agent.stop()

    def display_images(self, images):
        # Clear previous images
        for i in reversed(range(self.scroll_layout.count())): 
            self.scroll_layout.itemAt(i).widget().setParent(None)

        # Calculate grid dimensions
        n = len(images)
        cols = min(3, n)  # Max 3 columns
        rows = (n + cols - 1) // cols

        for i, img_array in enumerate(images):
            height, width, channel = img_array.shape
            bytes_per_line = 3 * width
            q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale down if too large
            if pixmap.width() > 300 or pixmap.height() > 300:
                pixmap = pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio)

            label = QLabel()
            label.setPixmap(pixmap)
            self.scroll_layout.addWidget(label, i // cols, i % cols)

    def closeEvent(self, event):

        if self.agent:
            self.close_agent()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CarControlUI()
    window.show()
    sys.exit(app.exec_())