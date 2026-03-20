import itertools

from src.s2c.experiments import ann_c_experiment
from src.s2c.utils import read_results

db_path = './data/data_points_filled.csv'
splits_path = './data/splitted_data.csv'
output_path = './data/ann_c/'


def main():
    df = read_results(output_path + 'ANN_C_acc.txt')

    param_grid = itertools.product(
        [64, 128, 256, 512],  # l1
        [32, 64, 128, 256],  # l2
        [16, 32, 64, 128],  # l3
        [0.001, 0.0001],  # lr
        [0.0, 0.15, 0.3],  # dropout
        [1, 2, 3],  # experiment
    )

    for l1, l2, l3, lr, dropout, experiment in param_grid:
        already_run = (
            (df['l1'] == l1) & (df['l2'] == l2) & (df['l3'] == l3)
            & (df['lr'] == lr) & (df['dropout'] == dropout) & (df['Experiment'] == experiment)
        ).any()

        if already_run:
            print(f'Skipping l1={l1}, l2={l2}, l3={l3}, lr={lr}, dropout={dropout}, experiment={experiment} because it has already been run')
            continue

        print(f'Running l1={l1}, l2={l2}, l3={l3}, lr={lr}, dropout={dropout}, experiment={experiment}')
        ann_c_experiment(
            db_path=db_path,
            splits_path=splits_path,
            experiment=experiment,
            output_path=output_path,
            l1=l1,
            l2=l2,
            l3=l3,
            dropout=dropout,
            epochs=500,
            lr=lr,
        )


if __name__ == '__main__':
    main()
