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
        env = StreetFighterCustomWrapper(env, reset_round=RESET_ROUND, rendering=False)
        return env
    return _init

game = "StreetFighterIISpecialChampionEdition-Genesis"
#state="Champion.Level12.RyuVsBison"
state = "test"
env = make_env(game, state)()
model = PPO.load(os.path.join(MODEL_DIR, MODEL_NAME), env=env)
ia = RetroInteractive(env, model)

ia.run()
# model = PPO("CnnPolicy", env)
'''
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
    
    if RESET_ROUND:
        obs = env.reset()

    total_reward = 0

    while not done:
        timestamp = time.time()

        if RANDOM_ACTION:
            #p1_action = env.get_player_agent(0).act(env.get_state()[0])
            #p2_action = env.get_player_agent(1).act(env.get_state()[1])
            print("action:" + str(env.action_space.sample()))
            obs, reward, done, info = env.step(env.action_space.sample())
        else:
            action, _states = model.predict(obs)
            print("action:" + str(action))
            two_actions = np.concatenate((action, action))
            print("two_actions:" + str(two_actions))
            obs, reward, done, info = env.step(two_actions)
            print("info:" + str(info))
        if reward != 0:
            total_reward += reward
            print("Reward: {:.3f}, playerHP: {}, enemyHP:{}".format(reward, info['agent_hp'], info['enemy_hp']))
        
        if info['enemy_hp'] < 0 or info['agent_hp'] < 0:
            done = True

    if info['enemy_hp'] < 0:
        print("Victory!")
        num_victory += 1

    print("Total reward: {}\n".format(total_reward))
    episode_reward_sum += total_reward

    if not RESET_ROUND:
        while info['enemy_hp'] < 0 or info['agent_hp'] < 0:
        # Inter scene transition. Do nothing.
            obs, reward, done, info = env.step([0] * 12)
            env.render()

env.close()
print("Winning rate: {}".format(1.0 * num_victory / num_episodes))
if RANDOM_ACTION:
    print("Average reward for random action: {}".format(episode_reward_sum/num_episodes))
else:
    print("Average reward for {}: {}".format(MODEL_NAME, episode_reward_sum/num_episodes))
    '''