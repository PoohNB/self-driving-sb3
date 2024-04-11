from collections import deque
from gym import spaces
import numpy as np




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
        self._skip_frame = skip_frame
        self.hist_len = hist_len
        self._act_num = act_num
        self.history_state = deque(maxlen=self.hist_len*(self._skip_frame+1)-self._skip_frame)

    def apply_action(self,act_num):
        self._act_num = act_num

    def gym_obs(self):

        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                                high=np.finfo(np.float32).max,
                                                shape=(1, (self.latent_space+self._act_num)*self.hist_len),
                                                dtype=np.float32)
        
        return self.observation_space
    
    def get_state(self):
        return np.concatenate([self.history_state[i] for i in range(len(self.history_state)) if i%(self._skip_frame+1)==0])
    
    def reset(self,img):
        seg_img = self.seg.predict(img)
        latent = self.vae.get_latent(seg_img)
        observation = np.concatenate((latent, np.zeros(self._act_num, np.float32)), axis=-1)
        self.history_state.extend([observation]*self.hist_len*(self._skip_frame+1)-self._skip_frame)
        
        return self.get_state
        
    def step(self,**arg):

        img = arg["img"]
        act = arg["act"]
        
        seg_img = self.seg.predict(img)
        latent = self.vae.get_latent(seg_img)
        observation = np.concatenate((latent, act), axis=-1)
        self.history_state.append(observation)

        return self.get_state
        

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
                 skip_frame=0):
        
        self.seg = seg_model
        self.vae = vae_model
        self.latent_space = latent_space
        self._skip_frame = skip_frame
        self.hist_len = hist_len
        self.history_state = deque(maxlen=self.hist_len*(self._skip_frame+1)-self._skip_frame)

    def apply_action(self,act_num):
        self._act_num = act_num

    def gym_obs(self):

        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                                high=np.finfo(np.float32).max,
                                                shape=(1, (self.latent_space)*self.hist_len),
                                                dtype=np.float32)
        
        return self.observation_space
    
    def get_state(self):
        return np.concatenate([self.history_state[i] for i in range(len(self.history_state)) if i%(self._skip_frame+1)==0])
    
    def reset(self,img):
        seg_img = self.seg.predict(img)
        observation = self.vae.get_latent(seg_img)
        self.history_state.extend([observation]*self.hist_len*(self._skip_frame+1)-self._skip_frame)
        
        return self.get_state
        
    def step(self,**arg):

        img = arg["img"]
        
        seg_img = self.seg.predict(img)
        observation = self.vae.get_latent(seg_img)
        self.history_state.append(observation)

        return self.get_state
        