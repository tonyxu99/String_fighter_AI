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

DEFAULT_MODEL_FILE = r"trained_models/ppo_ryu_2000000_steps_updated" # Speicify the model file to load. Model "ppo_ryu_2500000_steps_updated" is capable of beating the final stage (Bison) of the game.
#MODEL_NAME = r"ppo_ryu_10000000_steps"
# Model notes:
# ppo_ryu_2000000_steps_updated: Just beginning to overfit state, generalizable but not quite capable.
# ppo_ryu_2500000_steps_updated: Approaching the final overfitted state, cannot dominate first round but partially generalizable. High chance of beating the final stage.
# ppo_ryu_3000000_steps_updated: Near the final overfitted state, almost dominate first round but barely generalizable.
# ppo_ryu_7000000_steps_updated: Overfitted, dominates first round but not generalizable. 
DEFAULT_STATE = "Champion.Level1.RyuVsGuile"


def make_env(game, state, reset_type, rendering):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE
        )
        env = StreetFighterCustomWrapper(env, reset_type=reset_type, rendering=rendering)
        return env
    return _init

parser = argparse.ArgumentParser(description='Reset game stats')
parser.add_argument('--reset', choices=['round', 'match', 'game'], help='Reset stats for a round, a match, or the whole game', default='round')
parser.add_argument('--model-file', help='The model file to load. By default trained_models/ppo_ryu_2000000_steps_updated', default=DEFAULT_MODEL_FILE)
parser.add_argument('--state', help='The state file to load. By default Champion.Level1.RyuVsGuile', default=DEFAULT_STATE)
parser.add_argument('--skip-render', action='store_true', help='Whether to skip to render the game screen.')
parser.add_argument('--random-action', action='store_true', help='Use ramdom action instead of')
parser.add_argument('--num-episodes', type=int, help='Play how many episodes', default=30)

args = parser.parse_args()
reset_type = args.reset

print("command line args:" + str(args))

game = "StreetFighterIISpecialChampionEdition-Genesis"
#env = make_env(game, state="Champion.Level12.RyuVsBison")()
env = make_env(game, state=args.state, reset_type=args.reset, rendering=not args.skip_render)()
# model = PPO("CnnPolicy", env)

if not args.random_action:
    model = PPO.load(args.model_file, env=env)

obs = env.reset()
done = False

episode_reward_sum = 0
num_victory = 0

print("\nFighting Begins!\n")

for _ in range(args.num_episodes):
    done = False
    
    obs = env.reset()

    total_reward = 0

    while not done:
        timestamp = time.time()

        if args.random_action:
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
print("Winning rate: {}".format(1.0 * num_victory / args.num_episodes))
if args.random_action:
    print("Average reward for random action: {}".format(episode_reward_sum/args.num_episodes))
else:
    print("Average reward for {}: {}".format(args.model_file, episode_reward_sum/args.num_episodes))