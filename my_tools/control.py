import os
import cv2
import numpy as np

class ImageController:
    def __init__(self, process=None,folder = "data/raw_4cams", x_shape=1280, y_shape=720, box=(640, 417, 537, 255),max_idx = 4000):
        if process == None:
            self.process = self.crop_visual
            print("no function apply using default function")
        else:
            self.process = process
        self.folder = folder
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.cx, self.cy, self.w, self.h = box
        self.img_idx = 0
        self.cam_idx = 1
        self.max_idx = max_idx
        self.outputs = None

    def cam_call(self, x):
        self.cam_idx = x
        self.imgs = os.listdir(self.cams[self.cam_idx])
        self.imgs = sorted(self.imgs, key=lambda x: int(x.split(".")[0]))
        self.num_file = len(self.imgs)

    def img_call(self, x):
        self.img_idx = x

    def run(self):

        self.cams = [os.path.join(self.folder, item) for item in os.listdir(self.folder)]

        cv2.namedWindow("Trackbars")
        cv2.createTrackbar("cam", "Trackbars", self.cam_idx, len(self.cams) - 1, self.cam_call)
        cv2.createTrackbar("image", "Trackbars", self.img_idx, self.max_idx , self.img_call)
        cv2.createTrackbar("center x", "Trackbars", self.cx, self.x_shape, lambda x: None)
        cv2.createTrackbar("center y", "Trackbars", self.cy, self.y_shape, lambda x: None)
        cv2.createTrackbar("width", "Trackbars", self.w, int(self.x_shape / 2), lambda x: None)
        cv2.createTrackbar("height", "Trackbars", self.h, int(self.y_shape / 2), lambda x: None)

        cv2.resizeWindow("Trackbars", 640, 160)

        self.cam_call(self.cam_idx)

        try:
            while True:

                if self.img_idx >= self.num_file:
                    self.img_idx = self.num_file-1

                self.img = cv2.imread(os.path.join(self.cams[self.cam_idx], self.imgs[self.img_idx]))

                self.cx = cv2.getTrackbarPos("center x", "Trackbars")
                self.cy = cv2.getTrackbarPos("center y", "Trackbars")
                self.w = cv2.getTrackbarPos("width", "Trackbars")
                self.h = cv2.getTrackbarPos("height", "Trackbars")

                x1 = max(0, int(self.cx - self.w))
                y1 = max(0, int(self.cy - self.h))
                x2 = min(self.img.shape[1], int(self.cx + self.w))
                y2 = min(self.img.shape[0], int(self.cy + self.h))

                crop_box = (y1, y2, x1, x2)
                
                self.outputs = self.process(self.img, crop_box)

                for i,output in enumerate(self.outputs):
                    win_name = f"img{i}"
                    cv2.imshow(win_name, output)

                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    break

        except Exception as e:
            print(e)
        
        finally:

            cv2.destroyAllWindows()

      
    def crop_visual(self,img,crop_box):
        y1,y2, x1,x2 = crop_box
        mask = np.zeros_like(img)
        mask[y1:y2, x1:x2] = 255

        # Darken the areas outside the specified region
        img_darkened = cv2.addWeighted(img, 0.5, mask, 0.5, 0)

        return [img_darkened]
    
    def apply_processor(self,processor):
        self.process = processor

    def hlep(self):
        print(""" 
              
              processor input: image array 
                        output : list of image       
                                                      
              """)


