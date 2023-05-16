import os
import sys
import argparse

import retro
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from street_fighter_super_wrapper import StreetFighterSuperWrapper

LOG_DIR = 'logs'
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

def main():
    parser = argparse.ArgumentParser(description='Reset game stats')
    parser.add_argument('--reset', choices=['round', 'match', 'game'], help='Reset stats for a round, a match, or the whole game', default='round')
    parser.add_argument('--model-file', help='The model to continue to learn from.')
    parser.add_argument('--save-dir', help='The directory to save the trained models.', default = "trained_models")
    parser.add_argument('--model-name-prefix', help='The prefix of the model names to save.', default = "ppo_ryu")
    parser.add_argument('--state', help='The state file to load. By default Champion.Level1.RyuVsGuile', default=DEFAULT_STATE)
    parser.add_argument('--render', action='store_true', help='Whether to render the game screen.')
    parser.add_argument('--num-env', type=int, help='How many envirorments to create', default=16)
    parser.add_argument('--total-steps', type=int, help='How many total steps to train', default=20000000)
    parser.add_argument('--P2AI', action='store_true', help='AI control player 2.')
    
    args = parser.parse_args()

    print("command line args:" + str(args))
                                 
    # Set up the environment and model
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    env = SubprocVecEnv([make_env(game, state=args.state, reset_type=args.reset, rendering=args.render, p2ai=args.P2AI, seed=i) for i in range(args.num_env)])

    # Set linear schedule for learning rate
    # Start
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)

    # fine-tune
    # lr_schedule = linear_schedule(5.0e-5, 2.5e-6)

    # Set linear scheduler for clip range
    # Start
    clip_range_schedule = linear_schedule(0.15, 0.025)

    # fine-tune
    # clip_range_schedule = linear_schedule(0.075, 0.025)

    model = PPO(
        "CnnPolicy", 
        env,
        device="cuda", 
        verbose=1,
        n_steps=512,
        batch_size=512,
        n_epochs=4,
        gamma=0.94,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        tensorboard_log=LOG_DIR
    )

    if (args.model_file):
        print("load model from " + args.model_file)
        model.set_parameters(args.model_file)

    # Set the save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load the model from file
    # model_path = "trained_models/ppo_ryu_7000000_steps.zip"
    
    # Load model and modify the learning rate and entropy coefficient
    # custom_objects = {
    #     "learning_rate": lr_schedule,
    #     "clip_range": clip_range_schedule,
    #     "n_steps": 512
    # }
    # model = PPO.load(model_path, env=env, device="cuda", custom_objects=custom_objects)

    # Set up callbacks
    # Note that 1 timesetp = 6 frame
    checkpoint_interval = 31250 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=args.save_dir, name_prefix=args.model_name_prefix)

    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(args.save_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file
    
        model.learn(
            total_timesteps=args.total_steps, # total_timesteps = stage_interval * num_envs * num_stages (1120 rounds)
            callback=[checkpoint_callback]#, stage_increase_callback]
        )
        env.close()

    # Restore stdout
    sys.stdout = original_stdout

    # Save the final model
    model.save(os.path.join(args.save_dir, args.model_name_prefix + "_final.zip"))

if __name__ == "__main__":
    main()
