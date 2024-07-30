# import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QComboBox
# from PyQt5.QtGui import QPixmap, QImage
# from PyQt5.QtCore import Qt, QTimer
# import numpy as np

# class AutonomousCarUI(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Autonomous Car Control")
#         self.setGeometry(100, 100, 800, 600)

#         self.central_widget = QWidget()
#         self.setCentralWidget(self.central_widget)

#         self.layout = QHBoxLayout(self.central_widget)

#         # Control Panel
#         self.control_panel = QWidget()
#         self.control_layout = QVBoxLayout(self.control_panel)

#         self.forward_btn = QPushButton("Forward (0)")
#         self.left_btn = QPushButton("Left (1)")
#         self.right_btn = QPushButton("Right (2)")

#         self.control_layout.addWidget(self.forward_btn)
#         self.control_layout.addWidget(self.left_btn)
#         self.control_layout.addWidget(self.right_btn)

#         self.record_btn = QPushButton("Record")
#         self.replay_btn = QPushButton("Replay")
#         self.control_layout.addWidget(self.record_btn)
#         self.control_layout.addWidget(self.replay_btn)

#         # Image Display
#         self.image_display = QLabel()
#         self.image_display.setFixedSize(640, 480)

#         # Image Selector
#         self.image_selector = QComboBox()
#         self.image_selector.addItem("Select Image")

#         # Main Layout
#         self.layout.addWidget(self.control_panel)
#         self.layout.addWidget(self.image_display)
#         self.layout.addWidget(self.image_selector)

#         # Connect signals
#         self.forward_btn.clicked.connect(lambda: self.send_command(0))
#         self.left_btn.clicked.connect(lambda: self.send_command(1))
#         self.right_btn.clicked.connect(lambda: self.send_command(2))
#         self.record_btn.clicked.connect(self.toggle_recording)
#         self.replay_btn.clicked.connect(self.replay_recording)
#         self.image_selector.currentIndexChanged.connect(self.display_selected_image)

#         # Initialize variables
#         self.recording = False
#         self.recorded_commands = []
#         self.image_list = []  # This will store the numpy images

#     def send_command(self, command):
#         print(f"Sending command: {command}")
#         if self.recording:
#             self.recorded_commands.append(command)

#     def toggle_recording(self):
#         self.recording = not self.recording
#         self.record_btn.setText("Stop Recording" if self.recording else "Record")

#     def replay_recording(self):
#         print("Replaying recorded commands")
#         for command in self.recorded_commands:
#             self.send_command(command)
#             # You might want to add a delay here to simulate real-time replay

#     def add_image(self, np_image):
#         self.image_list.append(np_image)
#         self.image_selector.addItem(f"Image {len(self.image_list)}")

#     def display_selected_image(self, index):
#         if index > 0 and index <= len(self.image_list):
#             np_image = self.image_list[index - 1]
#             height, width, channel = np_image.shape
#             bytes_per_line = 3 * width
#             q_image = QImage(np_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
#             pixmap = QPixmap.fromImage(q_image)
#             self.image_display.setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = AutonomousCarUI()
#     window.show()
#     sys.exit(app.exec_())
# # print(getattr(SAC_trainer,"ENV_SAC"))

# # from config.Config_loader import get_train_config

# # print(get_train_config("SAC","SAC1","continuous"))

# # import argparse
# # import ast

# # def main():
# #     # Create the parser
# #     parser = argparse.ArgumentParser(description="Parse a dictionary from the command line")

# #     # Add an argument for the dictionary
# #     parser.add_argument('--dict', type=str, required=True, help="Dictionary in string format")

# #     # Parse the command-line arguments
# #     args = parser.parse_args()

# #     # Convert the string representation of the dictionary to an actual dictionary
# #     try:
# #         input_dict = ast.literal_eval(args.dict)
# #         if not isinstance(input_dict, dict):
# #             raise ValueError
# #     except (ValueError, SyntaxError):
# #         print("Error: The provided string is not a valid dictionary.")
# #         return

# #     # Use the dictionary
# #     print("Parsed dictionary:", input_dict)

# # if __name__ == "__main__":
# #     main()

from config import observer_config

# Function to get all attributes from a module
def get_filtered_attribute_names(module, prefix):
    attribute_names = [attr for attr in dir(module) if attr.startswith(prefix)]
    return attribute_names
print(get_filtered_attribute_names(observer_config,"observer"))
