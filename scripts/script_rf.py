import itertools

from src.s2c.experiments import rf_experiment
from src.s2c.utils import read_results

db_path = './data/data_points_filled.csv'
splits_path = './data/splitted_data.csv'
output_path = './data/rf/'

# Fixed args (single-element lists inlined)
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 2
MAX_SAMPLES = None


def main():
    df = read_results(output_path + 'rf_acc.txt')
    print('database loaded')

    param_grid = itertools.product(
        [50, 100, 200, 300, 400],  # n_estimators
        ["sqrt", "log2", None],  # max_features
        ['gini', 'entropy'],  # criterion
        [0.0, 0.01, 0.001],  # min_impurity_decrease
        [0.0, 0.01, 0.001],  # ccp_alpha
        [1, 2, 3],  # experiment
    )

    for n_estimators, max_features, criterion, min_impurity_decrease, ccp_alpha, experiment in param_grid:
        already_run = (
            (df['n_estimators'] == n_estimators)
            & (df['max_depth'] == (MAX_DEPTH if MAX_DEPTH is not None else 'None'))
            & (df['min_samples_split'] == MIN_SAMPLES_SPLIT)
            & (df['min_samples_leaf'] == MIN_SAMPLES_LEAF)
            & (df['max_features'] == (max_features if max_features is not None else 'None'))
            & (df['max_samples'] == (MAX_SAMPLES if MAX_SAMPLES is not None else 'None'))
            & (df['criterion'] == criterion)
            & (df['min_impurity_decrease'] == min_impurity_decrease)
            & (df['ccp_alpha'] == ccp_alpha)
            & (df['Experiment'] == experiment)
        ).any()

        if already_run:
            print(f'Skipping n_estimators={n_estimators}, max_features={max_features}, criterion={criterion}, min_impurity_decrease={min_impurity_decrease}, ccp_alpha={ccp_alpha}, experiment={experiment} because it has already been run')
            continue

        print(f'Running n_estimators={n_estimators}, max_features={max_features}, criterion={criterion}, min_impurity_decrease={min_impurity_decrease}, ccp_alpha={ccp_alpha}, experiment={experiment}')
        rf_experiment(
            db_path=db_path,
            splits_path=splits_path,
            experiment=experiment,
            output_path=output_path,
            n_estimators=n_estimators,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=max_features,
            random_state=42,
            max_samples=MAX_SAMPLES,
            criterion=criterion,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )

if __name__ == '__main__':
    main()
