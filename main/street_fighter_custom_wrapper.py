# Copyright 2023 LIN Yi. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
import time
import collections

import gym
import numpy as np

# Custom environment wrapper
class StreetFighterCustomWrapper(gym.Wrapper):
    def __init__(self, env, reset_type="round", rendering=False, step_extra_frame=True, p2ai = False):
        super(StreetFighterCustomWrapper, self).__init__(env)
        self.env = env

        self.p2ai = p2ai
        if p2ai:
            self.action_space = gym.spaces.MultiBinary(12)

        # Use a deque to store the last 9 frames
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        self.num_step_frames = 6

        self.reward_coeff = 3.0

        self.total_timesteps = 0

        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(100, 128, 3), dtype=np.uint8)
        
        self.step_extra_frame = step_extra_frame
        self.reset_type = reset_type
        self.rendering = rendering

        self.player_won = 0
        self.oppont_won = 0
        self.during_transation = False
    
    def _stack_observation(self):
        return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)

    def reset(self):
        observation = self.env.reset()
        
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.total_timesteps = 0
        
        # Clear the frame stack and add the first observation [num_frames] times
        self.frame_stack.clear()
        for _ in range(self.num_frames):
            self.frame_stack.append(observation[::2, ::2, :])

        return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)

    def step(self, action):
        custom_done = False
        if (self.p2ai):
            action = np.concatenate(([0] * 12, action))
        obs, _reward, _done, info = self.env.step(action)
        self.frame_stack.append(obs[::2, ::2, :])

        # Render the game if rendering flag is set to True.
        if self.rendering:
            self.env.render()
            time.sleep(0.01)

        if self.step_extra_frame:
            for _ in range(self.num_step_frames - 1):            
                # Keep the button pressed for (num_step_frames - 1) frames.
                obs, _reward, _done, info = self.env.step(action)
                self.frame_stack.append(obs[::2, ::2, :])
                if self.rendering:
                    self.env.render()
                    time.sleep(0.01)

        curr_player_health = info['agent_hp']
        curr_oppont_health = info['enemy_hp']
        timesup = info['round_countdown'] <=0  # Time's up

        self.total_timesteps += self.num_step_frames
        
        if (self.during_transation and (curr_player_health < 0 or curr_oppont_health < 0)):
            # During transation between episodes, do nothing
            custom_done = False
            custom_reward = 0
        else:
            self.during_transation = False 
            if (curr_player_health < 0 and curr_oppont_health < 0) or (timesup  and curr_player_health == curr_oppont_health):
                print ("Draw round")
                custom_reward = 1
                if (self.reset_type == "round"):
                    custom_done = True
                else:
                    custom_done = False
                    self.during_transation = True
            elif curr_player_health < 0 or (timesup and curr_player_health < curr_oppont_health):
                print("The round is over and player loses.")
                custom_reward = -math.pow(self.full_hp, (curr_oppont_health + 1) / (self.full_hp + 1))    # Use the remaining health points of opponent as penalty. 
                if (self.reset_type == "round"):
                    custom_done = True
                else:
                    self.during_transation = True
                    self.oppont_won += 1
                    if (self.oppont_won >= 2):
                        # Player loses the game
                        print("Player loses the game")
                        self.player_won = 0
                        self.oppont_won = 0
                        custom_done = not self.reset_type == "never"

            elif curr_oppont_health < 0 or (timesup and curr_player_health > curr_oppont_health):
                print("The round is over and player wins.")
                # custom_reward = curr_player_health * self.reward_coeff # Use the remaining health points of player as reward.
                                                                    # Multiply by reward_coeff to make the reward larger than the penalty to avoid cowardice of agent.

                # custom_reward = math.pow(self.full_hp, (5940 - self.total_timesteps) / 5940) * self.reward_coeff # Use the remaining time steps as reward.
                custom_reward = math.pow(self.full_hp, (curr_player_health + 1) / (self.full_hp + 1)) * self.reward_coeff

                if (self.reset_type == "round"):
                    custom_done = True
                else:
                    self.during_transation = True
                    self.player_won += 1
                    if (self.player_won >= 2):
                        # Player wins the match
                        print("Player wins the match")
                        self.player_won = 0
                        self.oppont_won = 0
                        custom_done = self.reset_type == "match"

            # While the fighting is still going on
            else:
                custom_reward = self.reward_coeff * (self.prev_oppont_health - curr_oppont_health) - (self.prev_player_health - curr_player_health)
                self.prev_player_health = curr_player_health
                self.prev_oppont_health = curr_oppont_health
                custom_done = False

        # if custom_reward != 0:
        #     print("reward:{}".format(custom_reward))

        # Max reward is 6 * full_hp = 1054 (damage * 3 + winning_reward * 3) norm_coefficient = 0.001
        return self._stack_observation(), 0.001 * custom_reward, custom_done, info # reward normalization
    