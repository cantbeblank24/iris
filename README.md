# Training World Models on Tokenized Action Sequences

This repository contains the code for our DL HS23 group project. We refer to
https://github.com/eloialonso/iris for an overview of the code base as well as how to
setup the environment. We had problems getting the codebase to work with newer Python
version and ended up using 3.8.

Data for results in sections 3.1 and 3.2 can be collected with the `collect.sh` script and
then plotted using `python3 plot_results.py results.csv`. We obtained the results in
section 3.3 by running the following commands (data is collected in WandB).


```
python src/main.py env.train.id=BattleZoneNoFrameskip-v4 common.extra_tokens=0
python src/main.py env.train.id=BattleZoneNoFrameskip-v4 common.extra_tokens=5
```

