from CARLAENV import CarlaEnv
from DRLMODEL import DDPG, OUNoise
from MEMORY import Memory

import pandas as pd
import time
import os
from shutil import copyfile
import copy
import numpy as np
import cv2

import torch

STEP = 1
REPLAY_SIZE = 8
PLAY_INTERVAL = 900
MAX_PLAY_STEP = 400

TEST_SCENE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

SEG_SIZE = 272
RGB_SIZE = 480

env = CarlaEnv()

estimator = DDPG()
noise = OUNoise()
memory = Memory()

df = pd.DataFrame(columns = ['DATE TIME', 'STEP', 'STEP REWARD', 'AVG PLAY SCORE', 'SUM PLAY SCORE',
                             'N PLAY STEP', 'BEST AVG PLAY SCORE', 'BEST SUM PLAY SCORE',
                             'N PLAY END EPISODE', 'SIGMA', 'TIMESTAMP'])

curr_timestamp = time.time()
curr_time = time.ctime(curr_timestamp)
curr_time = curr_time.replace(':', '-')
curr_time = curr_time.replace('  ', '_')
curr_time = curr_time.replace(' ', '_')

os.makedirs('save', exist_ok=True)
os.makedirs('memory', exist_ok=True)

path_save_name = f'save/{len(os.listdir("save"))+1}_[{curr_time}][{os.path.basename(__file__)[:-3]}]'
os.makedirs(f'{path_save_name}/best_all')
os.makedirs(f'{path_save_name}/best_sum')
os.makedirs(f'{path_save_name}/best_mean')

copyfile(os.path.basename(__file__), f'{path_save_name}/{os.path.basename(__file__)}')
copyfile('CARLAENV.py', f'{path_save_name}/CARLAENV.py')
copyfile('DRLMODEL.py', f'{path_save_name}/DRLMODEL.py')
copyfile('MEMORY.py', f'{path_save_name}/MEMORY.py')
copyfile('UTILS.py', f'{path_save_name}/UTILS.py')

step = copy.copy(STEP)
best_avg_play_score = -99999
best_sum_play_score = -99999

reward_step = []

estimator.actor_target.train()
estimator.critic.train()
estimator.critic_target.train()

def save_best(besttype):
    in_save_path = os.listdir(f'{path_save_name}/{besttype}')
    save_best_path = f'{path_save_name}/{besttype}/{len(in_save_path)}_{np.mean(play_reward):.6f}_{np.sum(play_reward):.6f}_{len(play_reward)}_{n_end}'
    os.makedirs(save_best_path)

    torch.save({
        'actor': estimator.actor.state_dict(),
        'actor_target': estimator.actor_target.state_dict(),
        'critic': estimator.critic.state_dict(),
        'critic_target': estimator.critic_target.state_dict(),
        'actor_optimizer': estimator.actor_optimizer.state_dict(),
        'critic_optimizer': estimator.critic_optimizer.state_dict()
        }, f'{save_best_path}/checkpoint.pth')
    
    writer = cv2.VideoWriter(f'{save_best_path}/play_rgb.mp4', 0x7634706d, 5, (env.rgb_width, RGB_SIZE))
    for n in range(env.frame_play):
        img = env.play_images[n]
        writer.write(img)
    writer.release()

    writer = cv2.VideoWriter(f'{save_best_path}/play_seg.mp4', 0x7634706d, 5, (SEG_SIZE, SEG_SIZE))
    for n in range(env.frame_play):
        img = np.zeros([SEG_SIZE, SEG_SIZE, 3], dtype=np.uint8)
        seg_convt = (env.play_images_seg[n]/3.0)*255
        img[:, :, 0] = seg_convt
        img[:, :, 1] = seg_convt
        img[:, :, 2] = seg_convt
        writer.write(img)
    writer.release()

    map_img_plot = copy.deepcopy(env.map_img)
    color_plot = {}

    for n in range(len(env.location_buffer)):
        if env.location_buffer[n][2] not in color_plot.keys():
            color_plot[env.location_buffer[n][2]] = np.random.randint(256, size=3)
        map_img_plot[env.location_buffer[n][1], env.location_buffer[n][0]] = color_plot[env.location_buffer[n][2]]
    cv2.imwrite(f'{save_best_path}/map_with_path.png', map_img_plot)

while True:
    env.reset()
    state = env.get_state()

    estimator.actor.train()

    while True:
        action = estimator.actor_predict_1(state).cpu().detach().numpy()[0]
        action = noise.get_action(action, step)

        reward, done, end, dont_record = env.step(action)

        action_cnvt = action.tolist()
        action_cnvt[0] = env.curr_steer_position

        env.action_state_buffer.append(action_cnvt)

        reward_step.append(reward)
        next_state = env.get_state()

        if not dont_record:
            memory.append(state, action, next_state, reward, done)

        if len(memory.memory) >= REPLAY_SIZE:
            replay_data = memory.sample()
            estimator.replay(replay_data)
        
        if step % PLAY_INTERVAL == 0:
            torch.save({
                'actor': estimator.actor.state_dict(),
                'actor_target': estimator.actor_target.state_dict(),
                'critic': estimator.critic.state_dict(),
                'critic_target': estimator.critic_target.state_dict(),
                'actor_optimizer': estimator.actor_optimizer.state_dict(),
                'critic_optimizer': estimator.critic_optimizer.state_dict()
                }, 'checkpoint.pth')
            
            env.move_to_restpoint()

            play_reward = []
            env.play = True
            n_end = 0

            estimator.actor.eval()
            for n in TEST_SCENE:
                n_play_step = 1
                env.reset(n)
                state = env.get_state()

                while True:
                    with torch.no_grad():
                        action = estimator.actor_predict_1(state).cpu().detach().numpy()[0]

                    reward, done, end, dont_record = env.step(action.tolist())
                    play_reward.append(reward)

                    if done or n_play_step>=MAX_PLAY_STEP-1:
                        env.move_to_restpoint()
                        if end:
                            n_end += 1
                        break

                    action_cnvt = action.tolist()
                    action_cnvt[0] = env.curr_steer_position
                    env.action_state_buffer.append(action_cnvt)
                    state = env.get_state()
                    n_play_step += 1

            curr_timestamp = time.time()
            curr_time = time.ctime(curr_timestamp)

            play_reward.append((len(TEST_SCENE)-n_end)*-500)

            print(f'[{curr_time}] STEP: {step}, STEP REWARD: {np.sum(reward_step):.6f}, ', end='')
            print(f'AVG PLAY SCORE: {np.mean(play_reward):.6f}, SUM PLAY SCORE: {np.sum(play_reward):.6f}, N PLAY STEP: {len(play_reward)}', end='')
            
            if n_end > 0:
                print(f' [PLAY END EPISODE {n_end}]', end='')

            if np.sum(play_reward) > best_sum_play_score:
                best_sum = True
                best_sum_play_score = np.sum(play_reward)
            else:
                best_sum = False
                
            if np.mean(play_reward) > best_avg_play_score:
                best_mean = True
                best_avg_play_score = np.mean(play_reward)
            else:
                best_mean = False

            save_data = {'DATE TIME': curr_time, 'STEP': step, 'STEP REWARD': np.sum(reward_step),
                         'AVG PLAY SCORE': np.mean(play_reward), 'SUM PLAY SCORE': np.sum(play_reward),
                         'N PLAY STEP': len(play_reward), 'BEST AVG PLAY SCORE': int(best_mean), 'BEST SUM PLAY SCORE': int(best_sum), 'N PLAY END EPISODE': n_end,
                         'SIGMA': noise.sigma, 'TIMESTAMP': curr_timestamp}
            
            df2 = pd.DataFrame([save_data.values()], columns=save_data.keys())
            df = pd.concat([df, df2], ignore_index=True)
            df.to_csv(f'{path_save_name}/history.csv', index=False)

            if best_sum and best_mean:
                print(f' [BEST ALL PLAY SCORE]')
                save_best('best_all')
            elif best_sum and not best_mean:
                print(f' [BEST SUM PLAY SCORE]')
                save_best('best_sum')
            elif not best_sum and best_mean:
                print(f' [BEST MEAN PLAY SCORE]')
                save_best('best_mean')
            else:
                print('')

            env.location_buffer = []
            env.frame_play = 0
            env.play_images = np.zeros([MAX_PLAY_STEP*len(TEST_SCENE)*2, RGB_SIZE, env.rgb_width, 3], dtype=np.uint8)
            env.play_images_seg = np.zeros([MAX_PLAY_STEP*len(TEST_SCENE)*2, SEG_SIZE, SEG_SIZE], dtype=np.uint8)

            reward_step = []
            env.play = False
            step += 1
            break

        step += 1

        if done:
            env.move_to_restpoint()
            break

        state = next_state
