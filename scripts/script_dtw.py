import itertools

from src.s2c.experiments import dtw_experiment, twdtw_experiment

db_path = './data/data_points_filled.csv'
splits_path = './data/splitted_data.csv'
output_path = './data/dtw/'


def main():
    for experiment in [1, 2, 3]:
        dtw_experiment(
            db_path=db_path,
            splits_path=splits_path,
            experiment=experiment,
            output_path=output_path,
        )

    twdtw_params = [
        (0.1, 100),
        (0.1, 50),
        (0.05, 35),
    ]
    for experiment, (alpha, beta) in itertools.product([1, 2, 3], twdtw_params):
        twdtw_experiment(
            db_path=db_path,
            splits_path=splits_path,
            experiment=experiment,
            output_path=output_path,
            alpha=alpha,
            beta=beta,
        )


if __name__ == '__main__':
    main()
