# Load model directly
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import numpy as np
from PIL import Image
import torch
from torch import nn




class HF_segmodel():
   
    def __init__(self,model_repo,custom_processor=None):

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("using ",self.device)

        self.custom_processor = custom_processor

        if self.custom_processor == None:
            self.processor = AutoImageProcessor.from_pretrained(model_repo)
        else:
            self.processor = self.custom_processor
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_repo).to(self.device)



    def predict(self,images,upsampling = False):

        """
        inputs: list of images
        return predict output and overlayed image
        """

        with torch.no_grad():
            if self.custom_processor == None:
                inputs = self.processor(images,return_tensors="pt")['pixel_values'].to(self.device)
            else:
                inputs = self.processor(images).unsqueeze(0).to(self.device)

            logits = self.model(pixel_values = inputs).logits

        if upsampling:

            upsampled_logits = nn.functional.interpolate(
                logits,
                size=images.shape[::-1], # (height, width)
                mode='bilinear',
                align_corners=False
            )

            pred_segs = upsampled_logits.argmax(dim=1).cpu()

            return pred_segs,upsampled_logits
        
        else:

            pred_segs = logits.argmax(dim=1).cpu()

            return pred_segs,logits
         
    def get_seg_overlay(self,images, segs):

        if len(images) != len(segs):
            raise Exception("number of images and seg result not equal")

        if images[0].shape != segs[0].shape:
            raise Exception("shape mismatch mack sure predict() have upsampling = True")

        imgs=[]
        for i in range(len(images)):
            color_seg = np.zeros((segs[i].shape[0], segs[i].shape[1], 3), dtype=np.uint8) # height, width, 3
            palette = np.array(self.sidewalk_palette())
            for label, color in enumerate(palette):
                color_seg[segs[i] == label, :] = color

            # Show image + mask
            img = np.array(images[i]) * 0.5 + color_seg * 0.5
            imgs.append(img.astype(np.uint8))

        return imgs
    
    def get_seg_image(self, segs):
        imgs=[]
        for i in range(len(segs)):
            color_seg = np.zeros((segs[i].shape[0], segs[i].shape[1], 3), dtype=np.uint8) # height, width, 3
            palette = np.array(self.sidewalk_palette())
            for label, color in enumerate(palette):
                color_seg[segs[i] == label, :] = color

            # Show image + mask
            img = color_seg * 0.5
            imgs.append(img.astype(np.uint8))

        return imgs

    def sidewalk_palette(self):
        """Sidewalk palette that maps each class to RGB values."""
        return [
            [155, 155, 155],
            [216, 82, 24],
            [255, 255, 0],
            [125, 46, 141],
            [118, 171, 47],
            [161, 19, 46],
            [255, 0, 0],
            [0, 128, 128],
            [190, 190, 0],
            [0, 255, 0],
            [0, 0, 255],
            [170, 0, 255],
            [84, 84, 0],
            [84, 170, 0],
            [84, 255, 0],
            [170, 84, 0],
            [170, 170, 0],
            [170, 255, 0],
            [255, 84, 0],
            [255, 170, 0],
            [255, 255, 0],
            [33, 138, 200],
            [0, 170, 127],
            [0, 255, 127],
            [84, 0, 127],
            [84, 84, 127],
            [84, 170, 127],
            [84, 255, 127],
            [170, 0, 127],
            [170, 84, 127],
            [170, 170, 127],
            [170, 255, 127],
            [255, 0, 127],
            [255, 84, 127],
            [255, 170, 127],
        ]