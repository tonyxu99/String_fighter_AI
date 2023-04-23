# String_fighter_AI

This project is folked from linyi's street-fighter-ai project. (https://github.com/linyiLYi/street-fighter-ai). Please refer the folked project's README for the environment setup.

My project has added the following features:

* Added a feature for human to play with the AI.

## Play with the AI

Find out the gym-retro game folder
```bash
python .\utils\print_game_lib_folder.py
```

Copy `Champion.Level12.RyuVsBison.2Player.state`, `Champion.Level12.RyuVsBison.state`, `data.json`, `metadata.json`, and `scenario.json` files from the `data/` folder file to gym-retro game folder. 

```bash
cd main
python play_with_ai.py
```

