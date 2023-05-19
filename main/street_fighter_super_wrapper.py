
import math
import time
import collections

import gym
import numpy as np

SUPER_ACTION = [
    [['FORWARD']], # Move forward
    [['BACKWARD']], # Move backward or defend   
    [['UP']], # Jump straight up    
    [['UP', 'FORWARD']], # Jump forward
    [['UP', 'BACKWARD']], # Jump backward
    [['DOWN', 'BACKWARD']], # Squat and defend
    [['A']],
    [['B']],
    [['C']],
    [['X']],
    [['Y']],
    [['Z']],
    [['None']],
    [['None'], ['None']],
    [['None'], ['None'], ['None']],
    [['DOWN'], ['DOWN', 'FORWARD'], ['FORWARD', 'X'], ['X']],  # 发波
    [['DOWN'], ['DOWN', 'BACKWARD'], ['BACKWARD', 'C'], ['C'], ['None'], ['None'], ['None']],  # 旋风腿
    [['FORWARD'], ['FORWARD'], ['DOWN'], ['DOWN', 'FORWARD'], ['FORWARD','X'], ['X'], ['None'], ['None'], ['None']],  # 升龙
]

# Custom environment wrapper
class StreetFighterSuperWrapper(gym.Wrapper):
    def __init__(self, env, reset_type="round", rendering=False, step_extra_frame=True, p2ai = False):
        super(StreetFighterSuperWrapper, self).__init__(env)
        self._env = env
        self._info = None

        self.p2ai = p2ai
        # if p2ai:
        #     self.action_space = gym.spaces.MultiBinary(12)
        self.action_space = gym.spaces.Discrete(len(SUPER_ACTION))

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

        self._empty_action = [0 for b in self._env.buttons]
    
    def _stack_observation(self):
        return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)

    def reset(self):
        observation = self._env.reset()
        
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.total_timesteps = 0
        
        # Clear the frame stack and add the first observation [num_frames] times
        self.frame_stack.clear()
        for _ in range(self.num_frames):
            self.frame_stack.append(observation[::2, ::2, :])

        return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)

    def _is_player_left_side(self):
        if (self._info == None):
            return not self.p2ai
        return self._info['agent_x'] < self._info['enemy_x']

    def _super_action_to_emu_action(self, super_action, seq_idx):
        keys = super_action[seq_idx]
        if 'FORWARD' in keys:
            index = keys.index('FORWARD')
            if self._is_player_left_side():
                keys[index] = 'RIGHT'
            else:
                keys[index] = 'LEFT'
        if 'BACKWARD' in keys:
            index = keys.index('BACKWARD')
            if self._is_player_left_side():
                keys[index] = 'LEFT'
            else:
                keys[index] = 'RIGHT'                
        return [b in keys for b in self._env.buttons]

    def step(self, super_action_idx):
        custom_done = False
        super_action = SUPER_ACTION[super_action_idx]
        action = self._super_action_to_emu_action(super_action, 0)
        if (self.p2ai):
            action = np.concatenate(([0] * 12, action))
        obs, _reward, _done, self._info = self._env.step(action)
        self.frame_stack.append(obs[::2, ::2, :])

        # Render the game if rendering flag is set to True.
        if self.rendering:
            self._env.render()
            time.sleep(0.01)

        if (len(super_action) > 1):
            for i in range(1, len(super_action) - 1):            
                action = self._super_action_to_emu_action(super_action, i)
                obs, _reward, _done, self._info = self._env.step(action)
                #print (self._info)
                self.frame_stack.append(obs[::2, ::2, :])
                if self.rendering:
                    self._env.render()
                    time.sleep(0.01)

            while (self._info["agent_status"] == 524):  # Doing special action
                obs, _reward, _done, self._info = self._env.step(self._empty_action)  # Do nothing if the agent is doing special action.
                #print("Doing special action. " + str(self._info["agent_status"]))
                self.frame_stack.append(obs[::2, ::2, :])
                if self.rendering:
                    self._env.render()
                    time.sleep(0.01)
            #print("The action is done. " + str(self._info["agent_status"]))

        curr_player_health = self._info['agent_hp']
        curr_oppont_health = self._info['enemy_hp']
        timesup = self._info['round_countdown'] <=0  # Time's up

        self.total_timesteps += self.num_step_frames
        
        if (self.during_transation and (curr_player_health < 0 or curr_oppont_health < 0)):
            # During transation between episodes, do nothing
            custom_done = False
            custom_reward = 0
        else:
            self.during_transation = False 
            if (curr_player_health < 0 and curr_oppont_health < 0) or (timesup  and curr_player_health == curr_oppont_health):
                #print ("Draw round")
                custom_reward = 1
                if (self.reset_type == "round"):
                    custom_done = True
                else:
                    custom_done = False
                    self.during_transation = True
            elif curr_player_health < 0 or (timesup and curr_player_health < curr_oppont_health):
                #print("The round is over and player loses.")
                custom_reward = -math.pow(self.full_hp, (curr_oppont_health + 1) / (self.full_hp + 1))    # Use the remaining health points of opponent as penalty. 
                if (self.reset_type == "round"):
                    custom_done = True
                else:
                    self.during_transation = True
                    self.oppont_won += 1
                    if (self.oppont_won >= 2):
                        # Player loses the game
                        #print("Player loses the game")
                        self.player_won = 0
                        self.oppont_won = 0
                        custom_done = not self.reset_type == "never"

            elif curr_oppont_health < 0 or (timesup and curr_player_health > curr_oppont_health):
                #print("The round is over and player wins.")
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
                        #print("Player wins the match")
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
        return self._stack_observation(), 0.001 * custom_reward, custom_done, self._info # reward normalization
    