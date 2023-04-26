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

import os
import time 
import argparse

import retro
from stable_baselines3 import PPO

from street_fighter_custom_wrapper import StreetFighterCustomWrapper

#RESET_ROUND = False  # Whether to reset the round when fight is over. 
RENDERING = True    # Whether to render the game screen.

#MODEL_NAME = r"ppo_ryu_2500000_steps_updated" # Speicify the model file to load. Model "ppo_ryu_2500000_steps_updated" is capable of beating the final stage (Bison) of the game.
MODEL_NAME = r"ppo_ryu_10000000_steps"
# Model notes:
# ppo_ryu_2000000_steps_updated: Just beginning to overfit state, generalizable but not quite capable.
# ppo_ryu_2500000_steps_updated: Approaching the final overfitted state, cannot dominate first round but partially generalizable. High chance of beating the final stage.
# ppo_ryu_3000000_steps_updated: Near the final overfitted state, almost dominate first round but barely generalizable.
# ppo_ryu_7000000_steps_updated: Overfitted, dominates first round but not generalizable. 

RANDOM_ACTION = False
NUM_EPISODES = 30
MODEL_DIR = r"trained_models_test/"

def make_env(game, state, reset_type):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE
        )
        env = StreetFighterCustomWrapper(env, reset_type=reset_type, rendering=RENDERING)
        return env
    return _init

parser = argparse.ArgumentParser(description='Reset game stats')
parser.add_argument('--reset', choices=['round', 'match', 'game'], help='Reset stats for a round, match, or game')

reset_type = "round"
args = parser.parse_args()
if args.reset == 'round':
    print('Resetting stats for a round...')
    reset_type = "round"
elif args.reset == 'match':
    print('Resetting stats for a match...')
    reset_type = "match"
elif args.reset == 'game':
    print('Resetting stats for a game...')
    reset_type = "game"
else:
    print('No reset option specified. Resetting stats for a round bydefault')
    reset_type = "round"

game = "StreetFighterIISpecialChampionEdition-Genesis"
#env = make_env(game, state="Champion.Level12.RyuVsBison")()
env = make_env(game, state="Champion.Level1.RyuVsGuile", reset_type=reset_type)()
# model = PPO("CnnPolicy", env)

if not RANDOM_ACTION:
    model = PPO.load(os.path.join(MODEL_DIR, MODEL_NAME), env=env)

obs = env.reset()
done = False

num_episodes = NUM_EPISODES
episode_reward_sum = 0
num_victory = 0

print("\nFighting Begins!\n")

for _ in range(num_episodes):
    done = False
    
    obs = env.reset()

    total_reward = 0

    while not done:
        timestamp = time.time()

        if RANDOM_ACTION:
            action = env.action_space.sample()
        else:
            action, _states = model.predict(obs)
        action[3] = 0 # Filter out the "START/PAUSE" button
        obs, reward, done, info = env.step(action)

        if reward != 0:
            total_reward += reward
            print("Reward: {:.3f}, playerHP: {}, enemyHP:{}".format(reward, info['agent_hp'], info['enemy_hp']))
        

    if info['enemy_hp'] < 0:
        print("Victory!")
        num_victory += 1

    print("Total reward: {}\n".format(total_reward))
    episode_reward_sum += total_reward


env.close()
print("Winning rate: {}".format(1.0 * num_victory / num_episodes))
if RANDOM_ACTION:
    print("Average reward for random action: {}".format(episode_reward_sum/num_episodes))
else:
    print("Average reward for {}: {}".format(MODEL_NAME, episode_reward_sum/num_episodes))