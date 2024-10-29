import serial
import cv2
import numpy as np 
import threading
import queue
from app_demo.agent import Agent
import time
import logging

def filter_8bit(steer,throttle):
    # Normalize steer to 0-255 and back to -1 to 1 with 8-bit precision
    new_steer = ((round(((steer+1)/2) * 255)/255)*2)-1
    # Normalize throttle to 0-255 and back to 0 to 1 with 8-bit precision
    new_throttle = (round(throttle*255)/255)

    return new_steer,new_throttle

class AgentSim:

    def __init__(self,model_path):

        self.agent = Agent(model_path)


        from environment.loader import env_from_config
        from config.Config_loader import get_env_config
        self.env = env_from_config(get_env_config(obs_module="observer_raw",
                                                    act_wrapper="action_original",
                                                    discrete_actions=None,
                                                    map_name="AIT",
                                                    level=-1)[0],
                                                    False)

    def reset(self):
        self.stopping = True
        self.images,_=self.env.reset()
        self.agent.reset(self.images)

    def get_vision(self):
            agent_vision = self.agent.render()
            return self.images+[img for list_img in agent_vision for img in list_img]

    def step(self,maneuver):
        action = self.agent(list_images=self.images,maneuver=maneuver)
        action = filter_8bit(action[0],action[1])
        if self.stopping:
            action = [0,0]
        self.images,_,_,_,_ =self.env.step(action)

    def start(self):
            self.stopping = False

    def stop(self):
        self.stopping = True

    def close(self):
        self.env.close()
        

class AgentReal:

    def __init__(self,
                 model_path,
                 n_cam):
        
        self.cams = RGBCams(n_cam)
        self.ctrl = SerialControl()
        self.agent = Agent(model_path)

    def reset(self):
        self.images = self.cams.read_images()
        self.agent.reset(self.images)
        self.ctrl.send(steer=0,throttle=0)

    def step(self,maneuver):
        self.images = self.cams.read_images()
        steer,throttle = self.agent(list_images=self.images,maneuver=maneuver)
        self.ctrl.send(steer=steer,throttle=throttle)

    def get_vision(self):
            agent_vision = self.agent.render()
            return self.images+[img for list_img in agent_vision for img in list_img]

    def stop(self):
        self.ctrl.send(steer=0,throttle=0,brake=True)
        time.sleep(2)
        self.ctrl.send(steer=0,throttle=0)

    def close(self):
        self.stop()
        self.ctrl.close()
        self.cams.close()




class RGBCams:
    def __init__(self, n_cam):
        self.caps = self.initialize_cameras(n_cam)

    def initialize_cameras(self, n_cam):
        caps = []
        for n in range(n_cam):  # Start from 0
            cap = cv2.VideoCapture(n, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print(f"Warning: Camera {n} failed to initialize")
            else:
                print(f"Camera {n} initialized")
            caps.append(cap)
        return caps

    def read_images(self):
        imgs = []
        for cap in self.caps:
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imgs.append(rgb_frame)
            else:
                print("Warning: Failed to read frame")
                imgs.append(np.zeros((480, 640, 3), dtype=np.uint8))  # Placeholder for failed capture
        self.rgb_image = imgs
        return self.rgb_image

    def close(self):
        for cap in self.caps:
            cap.release()



        
class SerialControl:

    def __init__(self,port="COM3",baud=115200):
        
        self.ser = self.initialize_serial_port(port, baud)

    def initialize_serial_port(self, port, baud):
        try:
            ser = serial.Serial(port, baud)
            print("Found car...")
            return ser
        except serial.SerialException as e:
            print(f"Serial error: {e}")
            print("turn to testing mode..")
            return None
        
    def send(self,steer,throttle,brake):

        if self.ser is not None:
            command = self.mc_cmd(steer, throttle, brake)
            self.ser.write(bytearray(command))
            self.ser.reset_input_buffer()
        else:
            logging.info(f"send dummy command: {command}")

    @staticmethod
    def mc_cmd(steer,throttle,brake=False):
        # steer -1,1
        scaled_steer= round(((steer +1)/2)*255) 

        # throttle 0,1
        scaled_throttle = round((throttle)*255)

        if brake:
            scaled_throttle = 0
            brake_signal = 255
        else:
            brake_signal = 0
            
        
        # [36,steer[0,255],throttle[0,255],brake[0 or 255],64]
        return [36,scaled_steer,scaled_throttle,brake_signal,64]
    
    def close(self):
        if self.ser is not None:
            self.ser.close()
            logging.info("Serial port closed.")





class Recorder:

    def __init__(self,video_path):
        self.recorded = queue.Queue()
        self.video_path = video_path
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = None
        self.is_recording = False
        self.saving_thread = threading.Thread(target=self.save_video_real_time)
        self.saving_thread.daemon = True
        self.saving_thread.start()
        self.rgb_image = []


    def add_images(self):
        if self.rgb_image:
            self.recorded.put(self.rgb_image)
        else:
            print("Warning: No images to add")

    def start_recording(self, fps=20.0):
        self.is_recording = True
        first_image_set = self.recorded.get()
        height, width, _ = first_image_set[0].shape
        self.recorded.task_done()
        n_cams = len(self.caps)
        grid_size = int(np.ceil(np.sqrt(n_cams)))
        frame_height = height // grid_size
        frame_width = width // grid_size
        output_height = frame_height * grid_size
        output_width = frame_width * grid_size

        self.out = cv2.VideoWriter(self.video_path, self.fourcc, fps, (output_width, output_height))

    def stop_recording(self):
        self.is_recording = False
        self.saving_thread.join()
        if self.out:
            self.out.release()
        print(f"Video saved at {self.video_path}")

    def save_video_real_time(self):
        while self.is_recording or not self.recorded.empty():
            if not self.recorded.empty():
                frame_set = self.recorded.get()
                height, width, _ = frame_set[0].shape
                n_cams = len(frame_set)
                grid_size = int(np.ceil(np.sqrt(n_cams)))
                frame_height = height // grid_size
                frame_width = width // grid_size
                output_height = frame_height * grid_size
                output_width = frame_width * grid_size

                combined_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                for i, frame in enumerate(frame_set):
                    resized_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (frame_width, frame_height))
                    row = i // grid_size
                    col = i % grid_size
                    combined_frame[row*frame_height:(row+1)*frame_height, col*frame_width:(col+1)*frame_width] = resized_frame

                self.out.write(combined_frame)
                self.recorded.task_done()