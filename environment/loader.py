# observer module (convert raw input to proper state)
# don't need vae_decoder if not reconstruct the latent
from environment.modules import observer

# action wrapper (object to post process the action like smooth, limit range or discretize)
from environment.modules import action_wrapper

# carla environment
from environment.CarlaEnv import CarlaImageEnv

def init_component(config):
    """
    Initialize components based on the configuration.

    Args:
        config (dict): Configuration dictionary with observer and action wrapper details.

    Returns:
        tuple: Initialized observer and action wrapper objects.
    """
    try:
        observer_class = getattr(observer, config['observer_config']['name'])
        observer_ob = observer_class(**config['observer_config']['config'])
    except (AttributeError, TypeError) as e:
        raise ValueError(f"Error initializing observer: {e}")

    try:
        action_wrapper_class = getattr(action_wrapper, config['actionwrapper']['name'])
        action_wrapper_ob = action_wrapper_class(**config['actionwrapper']['config'])
    except (AttributeError, TypeError) as e:
        raise ValueError(f"Error initializing action wrapper: {e}")

    return observer_ob, action_wrapper_ob

def env_from_config(config, RENDER):
    """
    Create environment instance from configuration.

    Args:
        config (dict): Configuration dictionary.
        RENDER (bool): Flag to activate rendering.

    Returns:
        CarlaImageEnv: Initialized CarlaImageEnv object.
    """
    obs, action_wr = init_component(config)

    print(config)
    env = CarlaImageEnv(
        observer=obs,
        action_wrapper=action_wr,
        activate_render=RENDER,
        render_raw=RENDER,
        render_observer=RENDER,
        **config['env_config']
    )

    return env
