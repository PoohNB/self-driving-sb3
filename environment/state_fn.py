
import torchvision.transforms as transforms
import torch

from autoencoder.encoder import VariationalEncoder
from autoencoder.reconstructor import Decoder

class VAE():

    def __init__(self,
                 latentdim=128,
                 encoder_path = None,
                 decoder_path = None):
        
        if encoder_path == None or decoder_path == None:
            raise Exception("not define model path")
        
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.latent_dim = latentdim
        self.load_model()

    def load_model(self):
        self.encoder = VariationalEncoder(latent_dims=self.latent_dim,modelpath = self.encoder_path)
        self.decoder = Decoder(latent_dims = self.latent_dim,modelpath = self.decoder_path)

    def preprocess_frame(self):
        preprocess = transforms.Compose([transforms.ToTensor()])
        frame = preprocess(frame).unsqueeze(0)
        return frame

    def encode_state(self,observe):
        encoded_state = {}
        with torch.no_grad():
            for seg in observe['seg']
            frame = self.preprocess_frame(seg)
            mu, logvar = self.vae.encode(frame)
            vae_latent = self.vae.reparameterize(mu, logvar)[0].cpu().detach().numpy().squeeze()
        encoded_state['vae_latent'] = vae_latent
        
        return encoded_state

    def reconstruct(self):
        pass

    def 
