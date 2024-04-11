import cv2
import numpy as np

class reward_from_map():

    def __init__(self,route_path):

        self.route = cv2.imread(route_path)

    def cart2pol(self, x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return rho, phi/np.pi*180

    def pol2cart(self, rho, phi):
        phi = phi/180*np.pi
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y
    

    def reward(self,car_pos,throttle,steer,velo):

        """
        it will get reward if 
        1. it on the mask and the front angle in the range

        penalty
        1. velocity very low (soft)
        2. car angle change too often in different direction (soft)
        3. out of mask (high)
        4. colision (high + depend on speed)

        car position =>  (x,y)
        angle => yaw

        """
        rew = 0


        self.route

        
        return rew
    
    


        # reward mask
        self.reward_mask = {}
        self.reward_mask['route_1.png'] = cv2.imread(f'../../utils/reward_mask/admin_to_aic/route_1.png')
        self.reward_mask['route_2.png'] = cv2.imread(f'../../utils/reward_mask/admin_to_aic/route_2.png')

        self.use_this_scene = ['route_1.png', 'route_2.png', 'route_1.png', 'route_2.png', 'route_1.png', 'route_1.png',
                               'route_2.png', 'route_2.png', 'route_2.png', 'route_1.png', 'route_1.png', 'route_2.png', 'route_2.png']

        # For checking the golf cart can hit road and lane marking
        self.check_seman_tags = [1, 24]



                # video 
        self.play = False
        self.play_images = np.zeros([MAX_PLAY_STEP*len(TEST_SCENE)*2, RGB_SIZE, self.rgb_width, 3], dtype=np.uint8)
        self.play_images_seg = np.zeros([MAX_PLAY_STEP*len(TEST_SCENE)*2, SEG_SIZE, SEG_SIZE], dtype=np.uint8)


        z



        self.location_buffer = []

        self.m = 8.596200822454035
        self.ref_point = (4005, 6864)

        self.seg_state_buffer = {s['name']:deque(maxlen=IN_CHANNLES) for s in self.camera_list}
        self.action_state_buffer = deque(maxlen=N_LOOK_BACK)

        self.frame_play = 0


            curr_pos = self.car.get_transform()
            x = round((curr_pos.location.x * self.m) + self.ref_point[1])
            y = round((curr_pos.location.y * self.m) + self.ref_point[0])

            if self.play:
                self.location_buffer.append([x, y, self.sp])
            
            yaw = -curr_pos.rotation.yaw
            img_car = np.ones((100, 100, 3), dtype=np.uint8)*255
            start_point = (round(1.43*self.m)+50, round(0.61*self.m)+50)
            end_point = (round(-1.73*self.m)+50, round(-0.61*self.m)+50)
            img_car = cv2.rectangle(img_car, start_point, end_point, (0, 0, 0), -1)
            img_car = ndimage.rotate(img_car, yaw, order=0, reshape=0, mode='nearest')

            img_car = cv2.bitwise_not(img_car[:, :, 0])/255
            img_road = self.reward_mask[self.use_this_scene[self.sp]][y-50:y+50, x-50:x+50, 0]/255
            try:
                bit_and = cv2.bitwise_and(img_car, img_road)
                dont_record = False
            except:
                dont_record = True
            
            if not dont_record:
                img_out_road = self.reward_mask[self.use_this_scene[self.sp]][y-50:y+50, x-50:x+50, 2]/255
                bit_out_road = cv2.bitwise_and(img_car, img_out_road)
                sum_all = np.sum(img_car)
                sum_and = np.sum(bit_and)
                sum_out = np.sum(bit_out_road)

                if self.sp in self.obs_location.keys():
                    curr_pos_obs = self.obs_car[self.which_obs].get_transform()

                    obs_range_mask = self.obs_class.curr_position(self.obstacles[self.which_obs][0], (curr_pos_obs.location.x, curr_pos_obs.location.y), curr_pos_obs.rotation.yaw)
                    obs_range_mask = obs_range_mask[y-50:y+50, x-50:x+50]/255
                    obs_bit_and = cv2.bitwise_and(img_car, obs_range_mask)
                    sum_mask = np.sum(obs_bit_and)

                velo = self.car.get_velocity()
                velo_r, velo_phi = self.cart2pol(velo.x, velo.y)
                convert_phi = yaw+velo_phi
                real_x, real_y = self.pol2cart(velo_r, convert_phi)
                real_x = real_x*3.6/8
                reward_real_x = (real_x-0.2)*3

                if self.sp in self.obs_location.keys():
                    if sum_mask >= 1:
                        reward -= 10
                        self.count_in_obs += 1
                    else:
                        reward += reward_real_x
                else:
                    reward += reward_real_x

                dis_end = []
                for n in range(len(self.end_points[self.sp])):
                    dis_end.append(curr_pos.location.distance(self.end_points[self.sp][n]))

                if curr_pos.location.distance(self.end_points[self.sp][2]) > self.prev_dist:
                    self.check_rev_dir.append(1)
                else:
                    self.check_rev_dir.append(0)

                self.prev_dist = curr_pos.location.distance(self.end_points[self.sp][2])

                if sum(self.check_rev_dir) >= N_CHECK_REVERSE:
                    end_this = True
                else:
                    end_this = False

                if sum_out > 0 or self.collision or end_this:
                    reward = -30
                    done = True
                    end = False

                elif min(dis_end) <= 2:
                    reward += ((sum_and/sum_all)-1)*1
                    if sum_and/sum_all < 0.98:
                        reward -= 1
                    done = True
                    end = True
                    
                else:
                    reward += ((sum_and/sum_all)-1)*1
                    if sum_and/sum_all < 0.98:
                        reward -= 1
                    done = False
                    end = False

                if done:
                    break

            else:
                break