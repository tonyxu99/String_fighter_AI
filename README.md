# String_fighter_AI

This project is folked from linyi's street-fighter-ai project. (https://github.com/linyiLYi/street-fighter-ai). Please refer the folked project's README for the environment setup.

My project has added the following features:

* Added a feature for human to [play aginst the AI](#play-aginst-ai).
* [Save state to a file](#save-state). 
* Allow human to press "START" button to join the fight (from 1 Player mode to 2 Player mode).
* Block AI to press "START" button as pressing "START" button will pause the game.
* Add [commmand line parameters](#commmand-line-parameters) to test.py and train.py
* [AI vs AI test](#ai-vs-ai-test)
* [Allow AI to play as player2](#player2_ai)
* AI vs AI training (comming soon)

## Play with the AI

### Setup
Find out the gym-retro game folder
```bash
python .\utils\print_game_lib_folder.py
```
Copy all files from the `data/` folder to gym-retro game folder. 

### <a name="play-aginst-ai"></a> Play aginst the AI
```bash
cd main
python play_with_ai.py
```

### Key Mapping
* Move - 'A', 'W', 'S', 'D'
* Arm - 'U', 'I', 'O'
* Leg - 'J', 'K', 'L'
* START (Join the fight / Pause) - 'N'
* Save state - 'B'

You can change the key mapping in keys_to_act() in interactive.py

### Trouble Shotting
You may see the following error 
```
Traceback (most recent call last):
  File "play_with_ai.py", line 41, in <module>
    model = PPO.load(os.path.join(MODEL_DIR, MODEL_NAME), env=env)
  File "/home/tony/AI/String_fighter_AI/venv/lib/python3.8/site-packages/stable_baselines3/common/base_class.py", line 684, in load
    check_for_correct_spaces(env, data["observation_space"], data["action_space"])
  File "/home/tony/AI/String_fighter_AI/venv/lib/python3.8/site-packages/stable_baselines3/common/utils.py", line 230, in check_for_correct_spaces
    raise ValueError(f"Action spaces do not match: {action_space} != {env.action_space}")
ValueError: Action spaces do not match: MultiBinary(12) != MultiBinary(24)
free(): invalid pointer
Aborted (core dumped)
```
This is because linyi's AI was trained with 1 player mode, so the AI expects 1 player action input. To work around the issue, just simplely comment out the action spaces validation in site-packages/stable_baselines3/common/utils.py (line 230).

## AI vs AI test
Copy `Champion.Level1.RyuVsRyu.2Player.state` files from the `data/` folder file to gym-retro game folder. 

```bash
cd main
python test_ai_vs_ai.py --reset=match
```
## <a name="player2_ai"></a>Train an AI to play as player 2 (right side)
1. Run play_with_ai.py
2. Press "N" to join the game as player2
3. Beat the player 1 (AI)
4. Press "B" to save the state. The state file will be save.state in the current directory
5. Modify the state file name as you like and copy to the retro game directory.
6. You can train the AI as P2:```python train.py --state <state name you saved> --P2AI --render```
7. Test the AI as P2:```python test.py --state <state name you saved> --P2AI ```

## <a name="save-state"></a>Save state

During playing with AI, you can press "B" button to save the state to a file (Champion.Level12.RyuVsBison.2Player.state）in the current directory. You can change the file name at line#124 in interactive.py

## Commmand Line Parameters
```
python test.py --help
usage: test.py [-h] [--reset {round,match,game}] [--model-file MODEL_FILE] [--state STATE] [--skip-render] [--random-action]
               [--num-episodes NUM_EPISODES]

Reset game stats

optional arguments:
  -h, --help            show this help message and exit
  --reset {round,match,game}
                        Reset stats for a round, a match, or the whole game
  --model-file MODEL_FILE
                        The model file to load. By default trained_models/ppo_ryu_2000000_steps_updated
  --state STATE         The state file to load. By default Champion.Level1.RyuVsGuile
  --skip-render         Whether to skip to render the game screen.
  --random-action       Use ramdom action instead of
  --num-episodes NUM_EPISODES
                        Play how many episodes
```
```
python train.py --help
usage: train.py [-h] [--reset {round,match,game}] [--model-file MODEL_FILE] [--save-dir SAVE_DIR] [--model-name-prefix MODEL_NAME_PREFIX]
                [--state STATE] [--render] [--num-env NUM_ENV] [--total-steps TOTAL_STEPS]

Reset game stats

optional arguments:
  -h, --help            show this help message and exit
  --reset {round,match,game}
                        Reset stats for a round, a match, or the whole game
  --model-file MODEL_FILE
                        The model to continue to learn from.
  --save-dir SAVE_DIR   The directory to save the trained models.
  --model-name-prefix MODEL_NAME_PREFIX
                        The prefix of the model names to save.
  --state STATE         The state file to load. By default Champion.Level1.RyuVsGuile
  --render              Whether to render the game screen.
  --num-env NUM_ENV     How many envirorments to create
  --total-steps TOTAL_STEPS
                        How many total steps to train
```
