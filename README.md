# Training World Models on Tokenized Action Sequences

This repository contains the code for our DL HS23 group project. We refer to
https://github.com/eloialonso/iris for an overview of the code base as well as how to
setup the environment. We had problems getting the codebase to work with newer Python
versions, 3.8.10 worked for us.

After setting up the environment according to https://github.com/eloialonso/iris we only
require that pretrained models are downloaded from
https://github.com/eloialonso/iris_pretrained_models.

## Code

https://github.com/cantbeblank24/iris/compare/main...final gives a pretty good overview of
our changes to the original iris code. These consist mainly of introducing the
tokenization logic and adding the necessary preprocessing step.

## Reproducing Results

Data for results in sections 3.1 and 3.2 can be collected with the `collect.sh` script and
then plotted using `python3 plot_results.py results.csv`.

Results from 3.3 can be obtained by running the following commands (scores are collected
on WandB).
```
python src/main.py env.train.id=BattleZoneNoFrameskip-v4 common.extra_tokens=5
# By setting extra_tokens to zero the original algorithm is retained
python src/main.py env.train.id=BattleZoneNoFrameskip-v4 common.extra_tokens=0
```

