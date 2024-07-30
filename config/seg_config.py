mask2former_repo = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
mask2former_labelmap = {13:1,7:1,23:2,24:2,52:3,55:3,57:3,20:4,21:4,22:4,19:4}

fbm2f_fp16 = dict(name = 'HF_mask2Formermodel',
                    config=dict(model_repo="facebook/mask2former-swin-large-mapillary-vistas-semantic",
                            fp16=True,
                            label_mapping=mask2former_labelmap,
                            crop=(512,1024))
                            )

fbm2f_fp16_1280 =dict(name = 'HF_mask2Formermodel',
                      config = dict(model_repo="facebook/mask2former-swin-large-mapillary-vistas-semantic",
                            fp16=True,
                            label_mapping=mask2former_labelmap,
                            crop=(640,1280))
                            )

fbm2f =dict(name = 'HF_mask2Formermodel',
            config = dict(model_repo="facebook/mask2former-swin-large-mapillary-vistas-semantic",
                            fp16=False,
                            label_mapping=mask2former_labelmap,
                            crop=(512,1024))
                            )