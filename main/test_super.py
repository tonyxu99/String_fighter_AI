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

import glob
import os
import time 
import argparse

import retro
from stable_baselines3 import PPO

from street_fighter_super_wrapper import StreetFighterSuperWrapper

DEFAULT_MODEL_FILE = r"trained_models/ppo_ryu_2000000_steps_updated" # Speicify the model file to load. Model "ppo_ryu_2500000_steps_updated" is capable of beating the final stage (Bison) of the game.
#MODEL_NAME = r"ppo_ryu_10000000_steps"
# Model notes:
# ppo_ryu_2000000_steps_updated: Just beginning to overfit state, generalizable but not quite capable.
# ppo_ryu_2500000_steps_updated: Approaching the final overfitted state, cannot dominate first round but partially generalizable. High chance of beating the final stage.
# ppo_ryu_3000000_steps_updated: Near the final overfitted state, almost dominate first round but barely generalizable.
# ppo_ryu_7000000_steps_updated: Overfitted, dominates first round but not generalizable. 
DEFAULT_STATE = "Champion.Level1.RyuVsGuile"


def make_env(game, state, reset_type, rendering, p2ai, verbose):
    def _init():
        players = 1
        if p2ai:
            players = 2
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            players=players
        )
        env = StreetFighterSuperWrapper(env, reset_type=reset_type, rendering=rendering, p2ai=p2ai, verbose=verbose)
        return env
    return _init

def list_model_files_in_directory(directory):
    files = glob.glob(directory + '/*.zip')
    files = [f for f in files if os.path.isfile(f)]
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reset game stats')
    parser.add_argument('--reset', choices=['round', 'match', 'game'], help='Reset stats for a round, a match, or the whole game', default='round')
    parser.add_argument('--model-file', help='The model file to load. By default trained_models/ppo_ryu_2000000_steps_updated', default=DEFAULT_MODEL_FILE)
    parser.add_argument('--model-dir', help='Test all model files in the dir', default="")
    parser.add_argument('--state', help='The state file to load. By default Champion.Level1.RyuVsGuile', default=DEFAULT_STATE)
    parser.add_argument('--skip-render', action='store_true', help='Whether to skip to render the game screen.')
    parser.add_argument('--verbose', action='store_true', help='Whether to display more information.')
    parser.add_argument('--num-episodes', type=int, help='Play how many episodes', default=10)
    parser.add_argument('--P2AI', action='store_true', help='AI control player 2.')

    args = parser.parse_args()

    print("command line args:" + str(args))

    game = "StreetFighterIISpecialChampionEdition-Genesis"
    #env = make_env(game, state="Champion.Level12.RyuVsBison")()
    env = make_env(game, state=args.state, reset_type=args.reset, rendering=not args.skip_render, p2ai=args.P2AI, verbose=args.verbose)()
    # model = PPO("CnnPolicy", env)

    if (args.model_dir != ""):
        model_files = list_model_files_in_directory(args.model_dir)
    else:
        model_files = [args.model_file]
    
    max_winning_rate_model = ""
    max_winning_rate = 0
    for model_file in model_files:
        model = PPO.load(model_file, env=env)

        obs = env.reset()
        done = False

        episode_reward_sum = 0
        total_player_won_matches = 0

        for _ in range(args.num_episodes):
            done = False
            
            obs = env.reset()

            total_reward = 0


            while not done:
                timestamp = time.time()

                action, _states = model.predict(obs)

                obs, reward, done, info = env.step(action)

                if reward != 0:
                    total_reward += reward
                    if args.verbose:            
                        print("Reward: {:.3f}, playerHP: {}, enemyHP:{}, player_won_matches:{}".format(reward, info['agent_hp'], info['enemy_hp'], info['player_won_matches']))
                
            total_player_won_matches += info['player_won_matches']

            if args.verbose:            
                print("Total player won matches: {}\n".format(total_player_won_matches))

        winning_rate = 1.0 * total_player_won_matches / args.num_episodes
        print("{} - winning rate: {}".format(model_file, winning_rate))
        if (winning_rate > max_winning_rate):
            max_winning_rate = winning_rate
            max_winning_rate_model = model_file
        if (max_winning_rate_model != ""):
            print("Max winning rate model: {} ({}) \n".format(max_winning_rate_model, max_winning_rate))

    env.close()

