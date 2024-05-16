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


class seg_vae_act():
    """
    class for convert image observation to state space

    

    """

    def __init__(self,
                 vae_model,
                 seg_model,
                 latent_space,
                 hist_len,
                 act_num=2,
                 skip_frame=0):
        
        self.seg = seg_model
        self.vae = vae_model
        self.latent_space = latent_space
        self.skip_frame = skip_frame
        self.hist_len = hist_len
        self.act_num = act_num
        self.history_state = deque(maxlen=self.hist_len*(self.skip_frame+1)-self.skip_frame)

    def gym_obs(self):

        observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                                high=np.finfo(np.float32).max,
                                                shape=(1, (self.latent_space+self.act_num)*self.hist_len),
                                                dtype=np.float32)
        
        return observation_space
    
    def get_state(self):
        return np.concatenate([self.history_state[i] for i in range(len(self.history_state)) if i%(self.skip_frame+1)==0])
    
    def reset(self,imgs):
        pred_segs = self.seg.predict(imgs)
        latents = self.vae.get_latent(pred_segs)
        cat_latent = torch.cat(latents, dim=1)
        observation = np.concatenate((cat_latent, [0]*self.act_num), axis=-1)
        self.history_state.extend(([observation]*(self.hist_len*(self.skip_frame+1)))-self.skip_frame)
        
        return self.get_state()
        
    def step(self,**arg):

        imgs = arg["imgs"]
        act = arg["act"]
        
        seg_imgs = self.seg.predict(imgs)
        latents = self.vae.get_latent(seg_imgs)
        cat_latent = torch.cat(latents, dim=1)
        observation = np.concatenate((cat_latent, act), axis=-1)
        self.history_state.append(observation)

        return self.get_state()

        

#==
class seg_vae():
    """
    class for convert image observation to state space

    """

    def __init__(self,
                 vae_model,
                 seg_model,
                 latent_space,
                 hist_len,
                 act_num=2,
                 skip_frame=0):
        
        self.seg = seg_model
        self.vae = vae_model
        self.latent_space = latent_space
        self.skip_frame = skip_frame
        self.hist_len = hist_len
        self.act_num = act_num
        self.history_state = deque(maxlen=self.hist_len*(self.skip_frame+1)-self.skip_frame)

    def apply_action(self,act_num):
        self.act_num = act_num

    def gym_obs(self):

        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                                high=np.finfo(np.float32).max,
                                                shape=(1, (self.latent_space+self.act_num)*self.hist_len),
                                                dtype=np.float32)
        
        return self.observation_space
    
    def get_state(self):
        return np.concatenate([self.history_state[i] for i in range(len(self.history_state)) if i%(self.skip_frame+1)==0])
    
    def reset(self,imgs):
        pred_segs = self.seg.predict(imgs)
        latents = self.vae.get_latent(pred_segs)
        cat_latent = torch.cat(latents, dim=1)
        observation = np.concatenate((cat_latent, [0]*self.act_num), axis=-1)
        self.history_state.extend(([observation]*(self.hist_len*(self.skip_frame+1)))-self.skip_frame)
        
        return self.get_state()
        
    def step(self,**arg):

        img = arg["img"]
        act = arg["act"]
        
        seg_img = self.seg.predict(img)
        latents = self.vae.get_latent(seg_img)
        observation = np.concatenate((latents, act), axis=-1)
        self.history_state.append(observation)

        return self.get_state()