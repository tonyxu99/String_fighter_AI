import os
import sys
import argparse
import traceback

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

def make_env(game, state, reset_type, rendering, p2ai = False, reward_coeff_base = 3.0, reward_coeff_coeff = 0.3, seed=0):
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
        env = StreetFighterSuperWrapper(env, rendering=rendering, reset_type=reset_type, p2ai=p2ai, 
                                        reward_coeff_base=reward_coeff_base, reward_coeff_coeff=reward_coeff_coeff)
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
        'n_envs': trail.suggest_int('n_envs', 4, 20, 4),
        'n_reward_coeff_base': trail.suggest_int('n_reward_coeff_base', 2, 5, 1),
        'n_reward_coeff_coeff': trail.suggest_float('n_reward_coeff_coeff', 0.1, 2),
    }

def make_optimize_agent(args):
    # Run a training loop and return mean reward 
    def optimize_agent(trial):
        try:
            model_params = optimize_ppo(trial) 
            n_envs = model_params['n_envs']
            reward_coeff_base = model_params['n_reward_coeff_base']
            reward_coeff_coeff = model_params['n_reward_coeff_coeff']
            del model_params['n_envs']
            del model_params['n_reward_coeff_base']
            del model_params['n_reward_coeff_coeff']
            
            # Create environment 
            game = "StreetFighterIISpecialChampionEdition-Genesis"
            env = SubprocVecEnv([make_env(game, state=args.state, reset_type=args.reset, rendering=args.render, 
                        reward_coeff_base=reward_coeff_base, reward_coeff_coeff=reward_coeff_coeff, seed=i) for i in range(n_envs)])

            # Create algo 
            model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **model_params)
            model.learn(total_timesteps=args.total_steps)

            print("start evaluate model")            
            # Evaluate model 
            total_player_won_matches = 0
            env = make_env(game, state=args.state, reset_type=args.reset, rendering=args.render, 
                        reward_coeff_base=reward_coeff_base, reward_coeff_coeff=reward_coeff_coeff)
            model.set_env(env)
            for _ in range(args.eval_episodes):
                done = False
                
                obs = env.reset()

                while not done:
                    action, _states = model.predict(obs)

                    obs, reward, done, info = env.step(action)

                total_player_won_matches += info['player_won_matches']

            print("Finished evaluate model. total_player_won_matches={}".format(total_player_won_matches))
            env.close()

            SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(trial.number))
            model.save(SAVE_PATH)

            return total_player_won_matches

        except Exception as e:
            # Capture the traceback as a string
            traceback_str = traceback.format_exc()
            # Do something with the traceback string
            print(traceback_str)
            return -1000
    return optimize_agent

def study_cb(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
    print("#### best_params={}".format(study.best_params))
    #print("best_trial={}".format(study.best_trial))

def main():
    parser = argparse.ArgumentParser(description='Reset game stats')
    parser.add_argument('--reset', choices=['round', 'match', 'game'], help='Reset stats for a round, a match, or the whole game', default='round')
    parser.add_argument('--state', help='The state file to load. By default Champion.Level1.RyuVsGuile', default=DEFAULT_STATE)
    parser.add_argument('--render', action='store_true', help='Whether to render the game screen.')
    parser.add_argument('--total-steps', type=int, help='How many total steps to train', default=100000)
    parser.add_argument('--opt-trials', type=int, help='How many optimization trials', default=100)
    parser.add_argument('--opt-jobs', type=int, help='How many trials to do in paralle', default=1)
    parser.add_argument('--eval-episodes', type=int, help='How many episodes to use to evaluate the trial', default=5)
    parser.add_argument('--study-name', help='Study Name')
    parser.add_argument('--study-storage', help='Study Storage')

    args = parser.parse_args()

    print("command line args:" + str(args))
                                 
    optimize_agent = make_optimize_agent(args)
    if (args.study_name != None and args.study_storage != None):
        # optuna create-study --study-name "ryu_opt" --storage "mysql+mysqldb://root@localhost/ryu" --direction maximize
        study = optuna.load_study(study_name=args.study_name, storage=args.study_storage)
    else:
        # Creating the experiment 
        study = optuna.create_study(direction='maximize')
    study.optimize(optimize_agent, n_trials=args.opt_trials, n_jobs=args.opt_jobs, gc_after_trial=True, callbacks=[study_cb])



if __name__ == "__main__":
    main()
