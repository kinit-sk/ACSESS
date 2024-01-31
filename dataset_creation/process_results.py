import os
import pandas as pd
import numpy as np
import pickle

# DATASET = 'DOG'
# DATASET = 'AWA'
# DATASET = 'APL'

# DATASET = 'ACT_410'
# DATASET = 'TEX_DTD'
# DATASET = 'PRT'
# DATASET = 'PNU'
# DATASET = 'FLW'

# DATASET = '20News'
# DATASET = 'NewsCategory'
# DATASET = 'atis'
# DATASET = 'facebook'
# DATASET = 'liu'
DATASET = 'snips'

# SUB_STRATEGY = 'cartography'
SUB_STRATEGY = None

path = os.path.join('..', 'meta-album', 'Code', 'Results', 'within_domain_fsl', DATASET)

with open('strategies_to_evaluate.pkl', 'rb') as file:
    strategies = pickle.load(file)

for shots in [5, 10]:
    shots_path = os.path.join(path, f'N5k{shots}test16')
    for model in ['protonet', 'maml', 'finetuning', 'mistral', 'zephyr']:
        model_path = os.path.join(shots_path, model)

        for strategy in strategies:
            if type(strategy) == list:
                strategy = '_'.join(strategy)
            strategy_path = os.path.join(model_path, f'{strategy}')
            if SUB_STRATEGY is not None:
                strategy_path = os.path.join(strategy_path, SUB_STRATEGY)

            results = pd.read_csv(os.path.join(strategy_path, f'{strategy}_selection_test_scores.csv'))
            score = results.mean_accuracy.tolist()

            mean = np.mean(score)
            std = np.std(score)

            print(f'Model: {model}, Shots: {shots}, Strategy: {strategy},{(mean * 100):.3f} +- {(std * 100):.3f}')
        print()
