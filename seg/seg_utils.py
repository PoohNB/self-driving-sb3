# Load model directly
from transformers import AutoImageProcessor,SegformerImageProcessor, SegformerForSemanticSegmentation,Mask2FormerForUniversalSegmentation
import numpy as np
from PIL import Image
import torch
from torch import nn
import cv2




class HF_segFormermodel():
   
    def __init__(self,model_repo,custom_processor=None):

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("using ",self.device)

        self.custom_processor = custom_processor

        if self.custom_processor == None:
            self.processor = SegformerImageProcessor.from_pretrained(model_repo)
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
                inputs = torch.stack([self.processor(img) for img in images],axis=0).to(self.device)

            logits = self.model(pixel_values = inputs).logits

        if upsampling:

            upsampled_logits = nn.functional.interpolate(
                logits,
                size=images[0].shape[:-1], # (height, width)
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

        imgs=[]
        for i in range(len(images)):
            color_seg = np.zeros((segs[i].shape[0], segs[i].shape[1], 3), dtype=np.uint8) # height, width, 3
            palette = np.array(self.sidewalk_palette())
            for label, color in enumerate(palette):
                color_seg[segs[i] == label, :] = color

            # Show image + mask
            if images[0].shape[:-1] != segs[0].shape:
                img = cv2.resize(images[i],(segs[i].shape[1],segs[i].shape[0])) * 0.5 + color_seg * 0.5
            else:
                img = images[i] * 0.5 + color_seg * 0.5

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
            img = color_seg 
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
    

# class HF_mask2formermodel():
   
#     def __init__(self,model_repo,custom_processor=None):

#         self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print("using ",self.device)

#         self.custom_processor = custom_processor


#         if self.custom_processor == None:
#             # self.processor = SegformerImageProcessor.from_pretrained(model_repo)
#             self.processor = AutoImageProcessor.from_pretrained(model_repo)
#         else:
#             self.processor = self.custom_processor
            
#         self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_repo).to(self.device)




#     def predict(self,images,upsampling = False):

#         """
#         inputs: list of images
#         return predict output and overlayed image
#         """

#         with torch.no_grad():
#             if self.custom_processor == None:
#                 inputs = self.processor(images,return_tensors="pt")['pixel_values'].to(self.device)
#             else:
#                 inputs = torch.stack([self.processor(img) for img in images],axis=0).to(self.device)

#             logits = self.model(pixel_values = inputs).logits

#         if upsampling:

#             upsampled_logits = nn.functional.interpolate(
#                 logits,
#                 size=images[0].shape[:-1], # (height, width)
#                 mode='bilinear',
#                 align_corners=False
#             )

#             pred_segs = upsampled_logits.argmax(dim=1).cpu()

#             return pred_segs,upsampled_logits
        
#         else:

#             pred_segs = logits.argmax(dim=1).cpu()

#             return pred_segs,logits
        

# tensorrt==================================================================================================
# convert the model in full-dimensions mode with an given input shape:
# /home/lpr/TensorRT-8.6.1.6/bin/trtexec \
# --onnx=fan-tiny/cityscapes_fan_tiny_hybrid_224.onnx \
# --saveEngine=fan-tiny/cityscapes_fan_tiny_hybrid_3x3x224x224_onnx_engine.trt \
# --shapes=input:3x3x224x224


import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit

import matplotlib.pyplot as plt
from PIL import Image

import tensorrt as trt
import os

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

# Check TensorRT version
print("TensorRT version:", trt.__version__)
assert trt.Builder(trt.Logger())































# note old version predict

# with self.engine.create_execution_context() as context:
#     # Set input shape based on image dimensions for inference
#     context.set_binding_shape(self.engine.get_binding_index("input"), (1, 3, 224,224))
#     # Allocate host and device buffers
#     bindings = []
#     for binding in self.engine:
#         binding_idx = self.engine.get_binding_index(binding)
#         size = trt.volume(context.get_binding_shape(binding_idx))
#         dtype = trt.nptype(self.engine.get_binding_dtype(binding))
#         if self.engine.binding_is_input(binding):
#             input_buffer = np.ascontiguousarray(preprocessed_image)
#             input_memory = cuda.mem_alloc(preprocessed_image.nbytes)
#             bindings.append(int(input_memory))
#         else:
#             output_buffer = cuda.pagelocked_empty(size, dtype)
#             output_memory = cuda.mem_alloc(output_buffer.nbytes)
#             bindings.append(int(output_memory))

#     stream = cuda.Stream()
#     # Transfer input data to the GPU.
#     cuda.memcpy_htod_async(input_memory, input_buffer, stream)
#     # Run inference
#     context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
#     # Transfer prediction output from the GPU.
#     cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
#     # Synchronize the stream
#     stream.synchronize()