

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



    def predict(self,image,give_overlay = False):

        """
        return predict output and overlayed image
        """

        if not isinstance(image,Image.Image):
            image = Image.fromarray(image)

        with torch.no_grad():
            if self.custom_processor == None:
                inputs = self.processor(image,return_tensors="pt")['pixel_values'].to(self.device)
            else:
                inputs = self.processor(image).unsqueeze(0).to(self.device)

            logits = self.model(pixel_values = inputs).logits

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1], # (height, width)
            mode='bilinear',
            align_corners=False
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu()

        if give_overlay:
            
            overlayed = self.get_seg_overlay(image,pred_seg)

        else:
            overlayed = None

        return pred_seg, overlayed   
         
    def get_seg_overlay(self,image, seg):
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
        palette = np.array(self.sidewalk_palette())
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color

        # Show image + mask
        img = np.array(image) * 0.5 + color_seg * 0.5
        img = img.astype(np.uint8)

        return img
    
    def get_seg_image(self, seg):
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
        palette = np.array(self.sidewalk_palette())
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color

        # Show image + mask
        img = color_seg
        img = img.astype(np.uint8)

        return img

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