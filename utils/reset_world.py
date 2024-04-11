import carla

CARLA_SERVER_IP = 'localhost'

client = carla.Client(CARLA_SERVER_IP, 2000)
client.set_timeout(10.0)
world = client.get_world()

settings = world.get_settings()

settings.synchronous_mode = False
settings.no_rendering_mode = False
settings.fixed_delta_seconds = 0
settings.substepping = True
settings.max_substep_delta_time = 0.01
settings.max_substeps = 10
settings.max_culling_distance = 0
settings.deterministic_ragdolls = True

world.apply_settings(settings)