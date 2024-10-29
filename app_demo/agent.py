import os
import pickle
from environment.loader import init_component
from config.algorithm_config import available_AlgorithmRL

class Agent:

    def __init__(self,model_path):
        model_dir = os.path.dirname(model_path)

        # load component
        config_path = os.path.join(model_dir, "env_config.pkl")
        with open(config_path, 'rb') as file:
            loaded_env_config = pickle.load(file)
        self.n_cam = len(loaded_env_config['env_config']['cam_config_list'])
        if "Vae" in loaded_env_config['observer_config']['name']:
            if loaded_env_config.get('observer_config', {}).get('config').get('vae_decoder_config') is None:
                vencoder_model_path = loaded_env_config['observer_config']['config']['vae_encoder_config']['model_path']
                vencoder_latent_dims = loaded_env_config['observer_config']['config']['vae_encoder_config']['latent_dims']
                decoder_model_path = os.path.join(os.path.dirname(vencoder_model_path), "decoder_model.pth")

                if os.path.exists(decoder_model_path):

                    loaded_env_config['observer_config']['config']['vae_decoder_config'] = {
                        'model_path': decoder_model_path,
                        'latent_dims': vencoder_latent_dims
                    }


        self.observer,self.action_wrapper=init_component(loaded_env_config)

        # load agent
        algo_name = model_path.split('/')[1].split('_')[0]
        algo = available_AlgorithmRL[algo_name]
        self.policy_net = algo.load(model_path, device='cuda')

    def reset(self,list_images):
        self.observer.reset(list_images)
        self.previous_action = [0,0]
    
    def render(self):
        displays = self.observer.get_renders()
        return displays

    def __call__(self,list_images,maneuver):

        """
        input: list_image:list of raw images
               action:agent predicted action
               maneuver: high level command 0,1,2 for guide the agent
        return: action (steer,throttle,brake)
        """

        obs = self.observer.step(
                            imgs = list_images,
                            act=self.previous_action,
                            maneuver=maneuver)
        self.previous_action,_ = self.policy_net.predict(obs.reshape((1,self.observer.len_latent)) ,deterministic=True)

        action = self.action_wrapper(self.previous_action)

        return action


