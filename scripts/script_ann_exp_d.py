import itertools

from src.s2c.experiments import ann_d_experiment
from src.s2c.utils import read_results

db_path = './data/data_points_filled.csv'
splits_path = './data/splitted_data.csv'
output_path = './data/ann_d/'


def main():
    df = read_results(output_path + 'ANN_D_acc.txt')

    param_grid = itertools.product(
        [128, 256, 512],  # dense1
        [128, 256, 512],  # dense2
        [32, 64, 128],  # conv1_channels
        [64, 128, 256],  # conv2_channels
        [128, 256, 512],  # conv3_channels
        [0.001, 0.0001],  # lr
        [0.0, 0.15, 0.3],  # dropout
        [1, 2, 3],  # experiment
    )

    for dense1, dense2, conv1_channels, conv2_channels, conv3_channels, lr, dropout, experiment in param_grid:
        already_run = (
            (df['dense1'] == dense1) & (df['dense2'] == dense2)
            & (df['conv1_channels'] == conv1_channels) & (df['conv2_channels'] == conv2_channels) & (df['conv3_channels'] == conv3_channels)
            & (df['lr'] == lr) & (df['dropout'] == dropout) & (df['Experiment'] == experiment)
        ).any()

        if already_run:
            print(f'Skipping dense1={dense1}, dense2={dense2}, conv1_channels={conv1_channels}, conv2_channels={conv2_channels}, conv3_channels={conv3_channels}, lr={lr}, dropout={dropout}, experiment={experiment} because it has already been run')
            continue

        print(f'Running dense1={dense1}, dense2={dense2}, conv1_channels={conv1_channels}, conv2_channels={conv2_channels}, conv3_channels={conv3_channels}, lr={lr}, dropout={dropout}, experiment={experiment}')
        ann_d_experiment(
            db_path=db_path,
            splits_path=splits_path,
            experiment=experiment,
            output_path=output_path,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            conv3_channels=conv3_channels,
            dense1=dense1,
            dense2=dense2,
            dropout=dropout,
            lr=lr,
            epochs=500,
        )


if __name__ == '__main__':
    main()
