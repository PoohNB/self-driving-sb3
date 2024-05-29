
# initial segmodel and encoder
from segmentation.seg_hf import HF_mask2Formermodel
from autoencoder.vae_wrapper import VencoderWrapper
from environment.tools.observer import SegVaeActHistObserver

modelrepo = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
mapping = {13:1,7:1,23:2,24:2,52:3,55:3,57:3,20:4,21:4,22:4,19:4}
seg_model = HF_mask2Formermodel(modelrepo,fp16=True,mapping=mapping)
#  # 0 is back ground
# seg_model.apply_label_mapping(mapping)

vae_encoder = VencoderWrapper(model_path="../autoencoder/model/vae32/best/var_encoder_model.pth",latent_dims=32)



observer = SegVaeActHistObserver(vae_encoder = vae_encoder,seg_model=seg_model,latent_space=32,hist_len = 8,skip_frame=0)
