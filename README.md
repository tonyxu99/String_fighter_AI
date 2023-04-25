# String_fighter_AI

This project is folked from linyi's street-fighter-ai project. (https://github.com/linyiLYi/street-fighter-ai). Please refer the folked project's README for the environment setup.

My project has added the following features:

* Added a feature for human to play with the AI.
* Save state to a file. 
* Allow human to press "START" button to join the fight (from 1 Player mode to 2 Player mode).
* Block AI to press "START" button as pressing "START" button will pause the game.

## Play with the AI

### Setup
Find out the gym-retro game folder
```bash
python .\utils\print_game_lib_folder.py
```
Copy `Champion.Level12.RyuVsBison.2Player.state`, `Champion.Level12.RyuVsBison.state`, `data.json`, `metadata.json`, and `scenario.json` files from the `data/` folder file to gym-retro game folder. 

### Play
```bash
cd main
python play_with_ai.py
```

### Key Mapping
* Move - 'A', 'W', 'S', 'D'
* Arm - 'U', 'I', 'O'
* Leg - 'J', 'K', 'L'

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

## Save state

During playing with AI, you can press "B" button to save the state to a file (Champion.Level12.RyuVsBison.2Player.state）in the current directory. You can change the file name at line#124 in interactive.py
