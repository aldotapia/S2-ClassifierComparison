import itertools

from src.s2c.experiments import ann_g_experiment
from src.s2c.utils import read_results

db_path = './data/data_points_filled.csv'
splits_path = './data/splitted_data.csv'
output_path = './data/ann_g/'


def main():
    df = read_results(output_path + 'ANN_G_acc.txt')

    param_grid = itertools.product(
        [128, 256],  # d_model
        [2, 4, 8, 16],  # num_heads
        [128, 256, 512],  # d_ff
        [1, 2, 4, 8],  # num_layers
        [0, 0.15, 0.3],  # dropout
        [0.001, 0.0001],  # lr
        [1, 2, 3],  # experiment
    )

    for d_model, num_heads, d_ff, num_layers, dropout, lr, experiment in param_grid:
        already_run = (
            (df['d_model'] == d_model) & (df['num_heads'] == num_heads) & (df['d_ff'] == d_ff)
            & (df['num_layers'] == num_layers) & (df['dropout'] == dropout) & (df['lr'] == lr) & (df['Experiment'] == experiment)
        ).any()

        if already_run:
            print(f'Skipping d_model={d_model}, num_heads={num_heads}, d_ff={d_ff}, num_layers={num_layers}, dropout={dropout}, lr={lr}, experiment={experiment} because it has already been run')
            continue

        print(f'Running d_model={d_model}, num_heads={num_heads}, d_ff={d_ff}, num_layers={num_layers}, dropout={dropout}, lr={lr}, experiment={experiment}')
        ann_g_experiment(
            db_path=db_path,
            splits_path=splits_path,
            experiment=experiment,
            output_path=output_path,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            epochs=500,
        )


if __name__ == '__main__':
    main()
