env_config = dict(host = 'localhost',
                  port = 2000,
                  vehicle = 'evt_echo_4s',
                  delta_frame=0.2,
                  check_reverse = 32)

ait_football_env = dict(**env_config,max_step =2000,)







