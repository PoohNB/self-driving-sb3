# observer is for convert the input from environment to the state

from collections import deque
from gymnasium import spaces
import numpy as np
import torch

class DummyObserver():
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
        self.len_latent = vae_encoder.latent_dims

    def get_gym_space(self):
        return  spaces.Box(low=np.finfo(np.float32).min,
                                                high=np.finfo(np.float32).max,
                                                shape=(1, self.len_latent),
                                                dtype=np.float32)
    
    
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
                 num_img_input,
                 act_num=2,
                 hist_len = 8,
                 skip_frame=0,
                 vae_decoder=None):
        
        super().__init__(seg_model=seg_model,
                         vae_encoder=vae_encoder,
                         vae_decoder=vae_decoder)

        self.latent_space = vae_encoder.latent_dims
        self.skip_frame = skip_frame
        self.hist_len = hist_len
        self.act_num = act_num
        self.len_latent = (self.latent_space*num_img_input+self.act_num)*self.hist_len
        self.history_state = deque(maxlen=self.hist_len*(self.skip_frame+1)-self.skip_frame)
    
    def get_state(self):
        state = np.concatenate([self.history_state[i] for i in range(len(self.history_state)) if i%(self.skip_frame+1)==0])
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
                 num_img_input,
                 act_num,
                 vae_decoder=None):
        
        super().__init__(seg_model=seg_model,
                         vae_encoder=vae_encoder,
                         vae_decoder=vae_decoder)

        self.latent_space = vae_encoder.latent_dims
        self.act_num = act_num
        self.len_latent = (self.latent_space+self.act_num)*num_img_input
    

        
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

class SegVaeActHistManvObserver(SegVaeObserver):

    def __init__(self,                 
                 seg_model,
                 vae_encoder,
                 num_img_input,
                 act_num=2,
                 maneuver_num=1,
                 hist_len = 8,
                 skip_frame=0,
                 vae_decoder=None):

        
        super().__init__(seg_model=seg_model,
                         vae_encoder=vae_encoder,
                         vae_decoder=vae_decoder)

        self.latent_space = vae_encoder.latent_dims
        self.skip_frame = skip_frame
        self.hist_len = hist_len
        self.act_num = act_num
        self.maneuver_num = maneuver_num
        self.len_latent = (self.latent_space*num_img_input+self.act_num+self.maneuver_num)*self.hist_len
        self.history_state = deque(maxlen=self.hist_len*(self.skip_frame+1)-self.skip_frame)

    def get_state(self):
        state = np.concatenate([self.history_state[i] for i in range(len(self.history_state)) if i%(self.skip_frame+1)==0])
        return state
    
    def reset(self,imgs):

        cat_latent = self.get_latent(imgs)
        observation = np.concatenate((cat_latent, [0]*self.act_num,[0]*self.maneuver_num), axis=-1)
        self.history_state.extend([observation]*((self.hist_len*(self.skip_frame+1))-self.skip_frame))

        return self.get_state()
        
    def step(self,**arg):

        imgs = arg["imgs"]
        act = arg["act"]
        manv= arg['maneuver']
        
        cat_latent = self.get_latent(imgs)
        observation = np.concatenate((cat_latent,act,manv), axis=-1)
        self.history_state.append(observation)

        return self.get_state()
        

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




observer_type = {'DummyObserver':DummyObserver,
                 'SegVaeObserver':SegVaeObserver,
                 'SegVaeActHistObserver':SegVaeActHistObserver,
                 'SegVaeActObserver':SegVaeActObserver,
                 'SegVaeActHistManvObserver':SegVaeActHistManvObserver,
                 'ClipObserver':ClipObserver}