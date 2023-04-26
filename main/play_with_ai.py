import os
import time
from interactive import RetroInteractive 

import retro
from stable_baselines3 import PPO
from street_fighter_custom_wrapper import StreetFighterCustomWrapper

RESET_ROUND = False  # Whether to reset the round when fight is over. 

MODEL_NAME = r"ppo_ryu_2500000_steps_updated" # Speicify the model file to load. Model "ppo_ryu_2500000_steps_updated" is capable of beating the final stage (Bison) of the game.

# Model notes:
# ppo_ryu_2000000_steps_updated: Just beginning to overfit state, generalizable but not quite capable.
# ppo_ryu_2500000_steps_updated: Approaching the final overfitted state, cannot dominate first round but partially generalizable. High chance of beating the final stage.
# ppo_ryu_3000000_steps_updated: Near the final overfitted state, almost dominate first round but barely generalizable.
# ppo_ryu_7000000_steps_updated: Overfitted, dominates first round but not generalizable. 

RANDOM_ACTION = False
NUM_EPISODES = 30 # Make sure NUM_EPISODES >= 3 if you set RESET_ROUND to False to see the whole final stage game.
MODEL_DIR = r"trained_models/"

def make_env(game, state):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            players=2
        )
        print("buttons:" + str(env.buttons))
        env = StreetFighterCustomWrapper(env, reset_type="game", rendering=False, step_extra_frame=False)
        return env
    return _init

game = "StreetFighterIISpecialChampionEdition-Genesis"
#state="Champion.Level12.RyuVsBison"
state = "Champion.Level12.RyuVsBison.2Player"
env = make_env(game, state)()
model = PPO.load(os.path.join(MODEL_DIR, MODEL_NAME), env=env)
ia = RetroInteractive(env, model)

ia.run()
