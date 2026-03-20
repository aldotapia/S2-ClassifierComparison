import itertools

from src.s2c.experiments import xgb_experiment
from src.s2c.utils import read_results

db_path = './data/data_points_filled.csv'
splits_path = './data/splitted_data.csv'
output_path = './data/xgb/'


def main():
    df = read_results(output_path + 'xgb_acc.txt')

    param_grid = itertools.product(
        [100, 500, 1000, 2000],  # n_estimators
        [3, 5, 9, 12],  # max_depth
        [1, 5, 10],  # min_child_weight
        [0.01, 0.1, 0.2],  # learning_rate
        [0.5, 0.7, 0.9, 1.0],  # subsample
        [0.5, 0.7, 0.9, 1.0],  # colsample_bytree
        [0.1, 1.0, 10.0],  # reg_lambda
        [0.0, 0.1, 1.0],  # reg_alpha
        [1, 2, 3],  # experiment
    )

    for n_estimators, max_depth, min_child_weight, lr, subsample, colsample_bytree, reg_lambda, reg_alpha, experiment in param_grid:
        already_run = (
            (df['n_estimators'] == n_estimators)
            & (df['max_depth'] == max_depth)
            & (df['min_child_weight'] == min_child_weight)
            & (df['learning_rate'] == lr)
            & (df['subsample'] == subsample)
            & (df['colsample_bytree'] == colsample_bytree)
            & (df['reg_lambda'] == reg_lambda)
            & (df['reg_alpha'] == reg_alpha)
            & (df['Experiment'] == experiment)
        ).any()

        if already_run:
            print(f'Skipping n_estimators={n_estimators}, max_depth={max_depth}, min_child_weight={min_child_weight}, learning_rate={lr}, subsample={subsample}, colsample_bytree={colsample_bytree}, reg_lambda={reg_lambda}, reg_alpha={reg_alpha}, experiment={experiment} because it has already been run')
            continue

        print(f'Running n_estimators={n_estimators}, max_depth={max_depth}, min_child_weight={min_child_weight}, learning_rate={lr}, subsample={subsample}, colsample_bytree={colsample_bytree}, reg_lambda={reg_lambda}, reg_alpha={reg_alpha}, experiment={experiment}')
        xgb_experiment(
            db_path=db_path,
            splits_path=splits_path,
            experiment=experiment,
            output_path=output_path,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            learning_rate=lr,
        )


if __name__ == '__main__':
    main()
