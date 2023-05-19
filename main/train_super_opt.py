import os
import sys
import argparse

import retro
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from street_fighter_super_wrapper import StreetFighterSuperWrapper

LOG_DIR = 'logs'
OPT_DIR = 'opts'
os.makedirs(LOG_DIR, exist_ok=True)

DEFAULT_STATE = "Champion.Level1.RyuVsGuile"
# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def make_env(game, state, reset_type, rendering, p2ai = False, seed=0):
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
        env = StreetFighterSuperWrapper(env, rendering=rendering, reset_type=reset_type, p2ai=p2ai)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init

def optimize_ppo(trail):
    return {
        'n_steps': trail.suggest_int('n_steps', 2048, 8192, 512),
        'batch_size': trail.suggest_int('batch_size', 64, 512, 64),
        'n_epochs': trail.suggest_int('n_epochs', 3, 8),
        'gamma': trail.suggest_float('gamma', 0.8, 0.99),
        'learning_rate': trail.suggest_float('learning_rate', 1e-5, 1e-4),
        'clip_range': trail.suggest_float('clip_range', 0.1, 0.4),
        'gae_lambda': trail.suggest_float('gae_lambda', 0.8, 0.99),
    }

def make_optimize_agent(args):
    # Run a training loop and return mean reward 
    def optimize_agent(trial):
        try:
            model_params = optimize_ppo(trial) 

            # Create environment 
            game = "StreetFighterIISpecialChampionEdition-Genesis"
            env = SubprocVecEnv([make_env(game, state=args.state, reset_type=args.reset, rendering=args.render, seed=i) for i in range(args.num_env)])

            # Create algo 
            model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **model_params)
            model.learn(total_timesteps=args.total_steps)

            print("start svaluate model")
            # Evaluate model 
            mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=2)
            print("Finished svaluate model. mean_reward={}".format(mean_reward))
            env.close()

            SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(trial.number))
            model.save(SAVE_PATH)

            return mean_reward

        except Exception as e:
            print (e)
            return -1000
    return optimize_agent
    
def main():
    parser = argparse.ArgumentParser(description='Reset game stats')
    parser.add_argument('--reset', choices=['round', 'match', 'game'], help='Reset stats for a round, a match, or the whole game', default='round')
    parser.add_argument('--state', help='The state file to load. By default Champion.Level1.RyuVsGuile', default=DEFAULT_STATE)
    parser.add_argument('--render', action='store_true', help='Whether to render the game screen.')
    parser.add_argument('--num-env', type=int, help='How many envirorments to create', default=16)
    parser.add_argument('--total-steps', type=int, help='How many total steps to train', default=100000)
    parser.add_argument('--opt-trials', type=int, help='How many optimization trials', default=100)

    args = parser.parse_args()

    print("command line args:" + str(args))
                                 
    optimize_agent = make_optimize_agent(args)
    # Creating the experiment 
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_agent, n_trials=args.opt_trials, n_jobs=1)

if __name__ == "__main__":
    main()
