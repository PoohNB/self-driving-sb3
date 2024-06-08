train_config1 = dict(NUM_EPOCHS = 100,
                BATCH_SIZE = 64,
                LEARNING_RATE = 1e-4,
                LATENT_SPACE = 24,
                SEED = 1234,)

vencoder16 =  dict(model_path="autoencoder/model/vae16/best/var_encoder_model.pth",
            latent_dims=16)
decoder16 = dict(model_path="autoencoder/model/vae16/best/decoder_model.pth",
            latent_dims=16)

vencoder32 = dict(model_path="autoencoder/model/vae32/best/var_encoder_model.pth",
            latent_dims=32)
decoder32 = dict(model_path="autoencoder/model/vae32/best/decoder_model.pth",
            latent_dims=32)

vencoder64 = dict(model_path="autoencoder/model/vae64/best/var_encoder_model.pth",
            latent_dims=64)

vencoder128 =  dict(model_path="autoencoder/model/vae128/best/var_encoder_model.pth",
            latent_dims=128)