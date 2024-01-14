import sys

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

df = pd.concat([
    pd.read_csv(path, header=None, names=['env', 'tokens', 'epochs', 'step', 'error'])
    for path in sys.argv[1:]
])

df['env'] = df['env'].str[:-14]
df = df[df['env'].isin([
    'Pong', 'Breakout', 'BattleZone', 'Boxing', 'Qbert', 'MsPacman'
])]

min_tokens = df.groupby('env').min()[['tokens']].rename(columns={'tokens': 'min_tokens'})
df = df.join(min_tokens, on='env')

df['tokens'] -= df['min_tokens']
df['approach'] = 'iris'
df['approach'][df['tokens'] > 0] = 'ours'

episode_end = df.iloc[:-1]['step'].values > df.iloc[1:]['step'].values
episode_end = np.append(episode_end, True)

mean_length = df[episode_end & (df['epochs'] == 215)].groupby(['env', 'tokens'])['step'].mean()
mean_length = mean_length.reset_index().pivot_table('step', ['env'], 'tokens')
mean_length = mean_length[[0, 4, 8]] / 20
mean_length = mean_length.style.format(precision=2)
print(mean_length.to_latex())

occurrences = df.groupby(
    ['env', 'step', 'tokens', 'epochs']
)[['error']].count().rename(
    columns={'error': 'count'}
)

df = df.join(occurrences, on=['env', 'step', 'tokens', 'epochs'])
df = df[df['count'] > 15]

df = df[(df['epochs'] == 215) & ((df['tokens'] == 0) | (df['tokens'] == 8))]

sns.relplot(
    data=df, x='step', y='error', hue='approach',
    palette="tab10",
    kind='line', col='env', col_wrap=3,
    # alpha=0.4,
    err_kws={'alpha': 0.4},
    facet_kws={'sharey': False, 'sharex': False},
)

plt.subplots_adjust(hspace=0.2)
plt.savefig("wm_results.png")
