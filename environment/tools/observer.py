# observer is for convert the input from environment to the state

from collections import deque
from gym import spaces
import numpy as np
import torch

class dummy_observer():
    """
    class for test only can't work with sb3

    """

    def __init__(self):
        
        pass

    def gym_obs(self):

        observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                                high=np.finfo(np.float32).max,
                                                shape=(1, 5),
                                                dtype=np.float32)
        
        return observation_space
    
    def get_state(self):
        return 
    
    def reset(self,imgs):

        
        return imgs
        
    def step(self,**arg):


        return arg["imgs"]
    
class SegVaeObserver():
    """
    class for convert image observation to state space

    seg_model = HFsegwrapper module
    vae_model = VencoderWrapper

    """


    def __init__(self,                
                seg_model,
                vae_encoder,
                vae_decoder=None):
        
        self.seg = seg_model
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder

    def gym_obs(self):

        raise NotImplementedError("Method 'gym_obs' must be implemented in subclasses")
    
    def reset(self,imgs):

        raise NotImplementedError("Method 'reset' must be implemented in subclasses")
        
    def step(self,**arg):

        raise NotImplementedError("Method 'step' must be implemented in subclasses")

    def get_latent(self,imgs):

        self.pred_segs = self.seg(imgs)
        self.latents = self.vae_encoder(self.pred_segs)
        cat_latent = self.latents.flatten().cpu().numpy()

        return cat_latent
    
    def get_renders(self):
        obsr = []
        obsr.append(self.seg.get_seg_images(self.pred_segs))
        if self.vae_decoder is not None:
            obsr.append(self.vae_decoder(self.latents))
        return obsr
    


class SegVaeActHistObserver(SegVaeObserver):
    """
    convert raw input to desired state

    """

    def __init__(self,                 
                 seg_model,
                 vae_encoder,
                 latent_space,
                 num_img_input,
                 act_num=2,
                 hist_len = 8,
                 skip_frame=0,
                 vae_decoder=None):
        
        super().__init__(seg_model=seg_model,
                         vae_encoder=vae_encoder,
                         vae_decoder=vae_decoder)

        self.latent_space = latent_space
        self.skip_frame = skip_frame
        self.hist_len = hist_len
        self.act_num = act_num
        self.len_latent = (self.latent_space*num_img_input+self.act_num)*self.hist_len
        self.history_state = deque(maxlen=self.hist_len*(self.skip_frame+1)-self.skip_frame)

    def gym_obs(self):

        observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                                high=np.finfo(np.float32).max,
                                                shape=(1, self.len_latent),
                                                dtype=np.float32)
        
        return observation_space
    
    def get_state(self):
        state = np.concatenate([self.history_state[i] for i in range(len(self.history_state)) if i%(self.skip_frame+1)==0])
        if state.shape[0] != self.len_latent:
            raise Exception("the state size and gym space not equal, please check observer argument 'num_img_input' and 'latent_space'")
        return state
    
    def reset(self,imgs):

        cat_latent = self.get_latent(imgs)
        observation = np.concatenate((cat_latent, [0]*self.act_num), axis=-1)
        self.history_state.extend([observation]*((self.hist_len*(self.skip_frame+1))-self.skip_frame))

        return self.get_state()
        
    def step(self,**arg):

        imgs = arg["imgs"]
        act = arg["act"]
        
        cat_latent = self.get_latent(imgs)
        observation = np.concatenate((cat_latent, act), axis=-1)
        self.history_state.append(observation)

        return self.get_state()

    

class SegVaeActObserver(SegVaeObserver):
    """
    work with carla environment as observation wrapper

    """

    def __init__(self,
                 vae_encoder,
                 seg_model,
                 observer_config,
                 vae_decoder=None):
        
        super().__init__(seg_model=seg_model,
                         vae_encoder=vae_encoder,
                         vae_decoder=vae_decoder)

        self.latent_space = observer_config['latent_space']
        self.act_num = observer_config['act_num']


    def gym_obs(self):

        observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                                high=np.finfo(np.float32).max,
                                                shape=(1, self.latent_space+self.act_num),
                                                dtype=np.float32)
        
        return observation_space
    
    def reset(self,imgs):

        cat_latent = self.get_latent(imgs)
        observation = np.concatenate((cat_latent, [0]*self.act_num), axis=-1)
        
        return observation
        
    def step(self,**arg):

        imgs = arg["imgs"]
        act = arg["act"]
        
        cat_latent = self.get_latent(imgs)
        observation = np.concatenate((cat_latent, act), axis=-1)

        return observation
    

#==
class SegVaeHistObserver():
    """
    class for convert image observation to state space
    seg_model = HFsegwrapper module
    vae_model = VencoderWrapper

    """

    def __init__(self,
                 vae_model,
                 seg_model,
                 latent_space,
                 hist_len,
                 skip_frame=0):
        
        self.seg = seg_model
        self.vae = vae_model
        self.latent_space = latent_space
        self.skip_frame = skip_frame
        self.hist_len = hist_len
        self.history_state = deque(maxlen=self.hist_len*(self.skip_frame+1)-self.skip_frame)


    def gym_obs(self):

        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                                high=np.finfo(np.float32).max,
                                                shape=(1, (self.latent_space)*self.hist_len),
                                                dtype=np.float32)
        
        return self.observation_space
    
    def get_state(self):
        return np.concatenate([self.history_state[i] for i in range(len(self.history_state)) if i%(self.skip_frame+1)==0])
    
    
    def reset(self,imgs):

        observation = self.get_latent(imgs)
        self.history_state.extend([observation]*((self.hist_len*(self.skip_frame+1))-self.skip_frame))
        
        return self.get_state()
        
    def step(self,**arg):

        imgs = arg["img"]
        
        observation = self.get_latent(imgs)
        self.history_state.append(observation)

        return self.get_state()
    
    def get_latent(self,imgs):

        pred_segs = self.seg(imgs)
        latents = self.vae(pred_segs)
        cat_latent = torch.cat(latents, dim=1)

        return cat_latent


class ClipObserver:

    def __init__(self,clip_model,split=3):
        self.split = split
        self.model = clip_model

    def gym_obs(self):

        pass

    def get_state(self):

        pass

    def reset(self):

        pass

    def step(self):

        pass

    def get_renders(self):
        return []
