import itertools

from src.s2c.experiments import ann_f_experiment
from src.s2c.utils import read_results

db_path = './data/data_points_filled.csv'
splits_path = './data/splitted_data.csv'
output_path = './data/ann_f/'


def main():
    df = read_results(output_path + 'ANN_F_acc.txt')

    param_grid = itertools.product(
        [0, 64, 128],  # projection_size
        [128, 256],  # dense1
        [128, 256],  # dense2
        [64, 128, 256],  # gru_hidden_size
        [1, 2, 3],  # gru_num_layers
        [0.001, 0.0001],  # lr
        [0.0, 0.15, 0.3],  # dropout
        [1, 2, 3],  # experiment
    )

    for projection_size, dense1, dense2, gru_hidden_size, gru_num_layers, lr, dropout, experiment in param_grid:
        already_run = (
            (df['projection_size'] == projection_size) & (df['dense1'] == dense1) & (df['dense2'] == dense2)
            & (df['gru_hidden_size'] == gru_hidden_size) & (df['gru_num_layers'] == gru_num_layers)
            & (df['lr'] == lr) & (df['dropout'] == dropout) & (df['Experiment'] == experiment)
        ).any()

        if already_run:
            print(f'Skipping projection_size={projection_size}, dense1={dense1}, dense2={dense2}, gru_hidden_size={gru_hidden_size}, gru_num_layers={gru_num_layers}, lr={lr}, dropout={dropout}, experiment={experiment} because it has already been run')
            continue

        print(f'Running projection_size={projection_size}, dense1={dense1}, dense2={dense2}, gru_hidden_size={gru_hidden_size}, gru_num_layers={gru_num_layers}, lr={lr}, dropout={dropout}, experiment={experiment}')
        ann_f_experiment(
            db_path=db_path,
            splits_path=splits_path,
            experiment=experiment,
            output_path=output_path,
            projection_size=projection_size,
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
            dense1=dense1,
            dense2=dense2,
            dropout=dropout,
            lr=lr,
            epochs=500,
        )


if __name__ == '__main__':
    main()
