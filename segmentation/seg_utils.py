# Load model directly
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation,AutoImageProcessor,Mask2FormerForUniversalSegmentation
import numpy as np
from PIL import Image
import torch
from torch import nn
import cv2
import colorsys
import time

class SegmodelWrapper():

    """
    this class consist of method like post process to manipulate the label result, 
                                    generate colors for each class,
                                    give the overlayed result
    requirement for this class

    init
    self.default_numlabels:
    have to use apply_label_mapping(label_mapping,custom_palette) after init materials for predict method
        
        
    method:

        warmup(self,image_shape=(512,1024,3)) : do inference and print the avg time
        generate_colors(num_classes) : generate color for labels
        apply_label_mapping(self,label_mapping,custom_palette = None): it apply the label mapping for post process 
        predict(self,images): have to imprement
        convert_label(self,seg_maps): for convert original semantic map to desired semantic map base on label_mapping
        get_seg_overlay(self,images, segs,original_size = True): get the image overlay results

    """

   
    def __init__(self):        

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("using ",self.device)


    def warmup(self,image_shape=(512,1024,3),batch=1):
        # warmup
        dummpy_images = [np.random.randint(0, 255, image_shape, dtype=np.uint8)]*batch
        self.predict(dummpy_images)
        st = time.time()
        for _ in range(3):
            self.predict(dummpy_images)
        print(f"inference time :{(time.time()-st)/3:.2f}")

    @staticmethod
    def generate_colors(num_classes):
        hsv_colors = [(i / num_classes, 1.0, 1.0) for i in range(num_classes)]
        rgb_colors = [colorsys.hsv_to_rgb(*color) for color in hsv_colors]
        rgb_colors = [(int(r * 255), int(g * 255), int(b * 255)) for (r, g, b) in rgb_colors]
        return np.array(rgb_colors).astype(np.uint8)
    
    def apply_label_mapping(self,label_mapping,custom_palette = None):
        
        if label_mapping is not None:
            self.labels = list(set(label_mapping.values()))
            assert 0 not in self.labels, "0 is background, label can't be 0"
            num_labels = len(self.labels)
        else: 
            self.labels = list(range(self.default_numlabels))
            num_labels = self.default_numlabels
        self.label_mapping = label_mapping
        self.palette = custom_palette if custom_palette and len(custom_palette) >= num_labels \
            else self.generate_colors(num_labels)


    def predict(self,images):

        """
        inputs: list of images
        return predict output and overlayed image

        should apply for post process
        if self.label_mapping is not None:
            pred_segs = self.convert_label(pred_segs)  

        """
        raise NotImplementedError("Method 'predict' must be implemented in subclasses")
    
    def convert_label(self,seg_maps):

        new_segmaps = []

        for seg_map in seg_maps:

            # postprocessed = np.full(seg_map.shape,255, dtype=np.uint8)
            postprocessed = np.zeros(seg_map.shape, dtype=np.uint8)

            for k,v in self.label_mapping.items():
                postprocessed[seg_map==k]=v

            new_segmaps.append(postprocessed)

        return new_segmaps
         
    
    def get_seg_overlay(self,images, segs,original_size = True):

        if len(images) != len(segs):
            raise ValueError("Number of images and segmentation results must be equal")
        
        imgs=[]
        for i in range(len(images)):
            color_seg = np.zeros((segs[i].shape[0], segs[i].shape[1], 3), dtype=np.uint8) # height, width, 3
            for j, label in enumerate(self.labels):
                color_seg[segs[i] == label, :] = self.palette[j]

            color_seg[segs[i] == 0, :] = [0,0,0]

            # Show image + mask
            if images[0].shape[:-1] != segs[0].shape:
                if original_size:
                    img = images[i] * 0.5 + cv2.resize(color_seg,(images[i].shape[1],images[i].shape[0]),interpolation= cv2.INTER_LINEAR) * 0.5
                else:
                    img = cv2.resize(images[i],(segs[i].shape[1],segs[i].shape[0])) * 0.5 + color_seg * 0.5
            else:
                img = images[i] * 0.5 + color_seg * 0.5

            imgs.append(img.astype(np.uint8))

        return imgs
    
    def get_seg_images(self, segs,shape =None):

 
        imgs=[]
        for i in range(len(segs)):
            color_seg = np.zeros((segs[i].shape[0], segs[i].shape[1], 3), dtype=np.uint8) # height, width, 3
            for j, label in enumerate(self.labels):
                color_seg[segs[i] == label, :] = self.palette[j]

            color_seg[segs[i] == 0, :] = [0,0,0]

            if shape is not None:
                img = cv2.resize(color_seg,shape)
            else:
                img = color_seg

            imgs.append(img.astype(np.uint8))

        return imgs
    

