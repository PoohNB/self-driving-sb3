
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation,AutoImageProcessor,Mask2FormerForUniversalSegmentation
import torch
import numpy as np
from PIL import Image
import torch
from torch import nn
import cv2

from seg.seg_utils import SegmodelWrapper

    
class HF_segFormermodel(SegmodelWrapper):
   
    def __init__(self,model_repo,label_mapping=None,custom_processor=None,custom_palette=None):

        super().__init__()

        self.label_mapping = label_mapping
        self.custom_processor = custom_processor
        self.processor = custom_processor or SegformerImageProcessor.from_pretrained(model_repo)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_repo).to(self.device)
        # self.model = torch.compile(self.model)
        self.default_numlabels = self.model.config.num_labels
        self.apply_label_mapping(label_mapping,custom_palette)
        self.warmup()
    
    def predict(self,images,upsampling = False):

        """
        inputs: list of images
        return predict output and overlayed image
        """

        with torch.no_grad():
            if self.custom_processor == None:
                inputs = self.processor(images,return_tensors="pt")['pixel_values'].to(self.device)
            else:
                inputs = self.processor(torch.stack([self.processor(img) for img in images],axis=0)).to(self.device)

            logits = self.model(pixel_values = inputs).logits

        if upsampling:

            logits = nn.functional.interpolate(
                logits,
                size=images[0].shape[:-1], # (height, width)
                mode='bilinear',
                align_corners=False
            )

            pred_segs = logits.argmax(dim=1).cpu()
        
        else:

            pred_segs = logits.argmax(dim=1).cpu()

        if self.label_mapping is not None:
            pred_segs = self.convert_label(pred_segs)  

        return pred_segs,logits
         
    

class HF_mask2Formermodel(SegmodelWrapper):
   
    def __init__(self,model_repo,label_mapping = None,custom_processor=None,custom_palette=None):
        super().__init__()

        if label_mapping is not None:
            assert isinstance(label_mapping,dict)

        self.custom_processor = custom_processor
        self.processor = AutoImageProcessor.from_pretrained(model_repo) if self.custom_processor is None else self.custom_processor
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_repo).to(self.device)
        # self.model = torch.compile(self.model)
        self.default_numlabels = self.model.config.num_labels
        self.apply_label_mapping(label_mapping,custom_palette)
        self.warmup()

    def predict(self,images):

        """
        inputs: list of images
        return predict output and overlayed image
        """

        with torch.no_grad():
            if self.custom_processor == None:
                inputs = self.processor(images,return_tensors="pt")['pixel_values'].to(self.device)
            else:
                inputs = self.processor(torch.stack([self.processor(img) for img in images],axis=0)).to(self.device)

            outputs = self.model(pixel_values = inputs)

        seg_maps = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[image.shape[:-1] for image in images])

        seg_maps = [seg.cpu().numpy() for seg in seg_maps]

        if self.label_mapping is not None:
            seg_maps = self.convert_label(seg_maps)

        return seg_maps,outputs
    



