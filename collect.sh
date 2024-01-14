set -e

declare -a envs=(Breakout Qbert Pong Boxing MsPacman BattleZone)

epochs=215

for env in "${envs[@]}"
do
    echo "Copying checkpoint"
    cp -f iris_pretrained_models/pretrained_models/${env}.pt checkpoints/last.pt

    for n_tokens in "${tokens[@]}"
    do
        echo "Running experiment for env: ${env}, n_tokens: ${n_tokens}"

        python3 src/main.py \
            env.train.id=${env}NoFrameskip-v4 \
            common.epochs=${epochs} \
            common.extra_tokens=${n_tokens} \
            collection.test.config.temperature=1.0 \
            evaluation.every=${epochs} \
            wandb.mode=offline
        cat $(ls -t outputs/**/**/results.csv | head -n 1) >> results.csv
    done
done

echo "Stored results in result.csv, you can plot them with the plot_results.py script"


