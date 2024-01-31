import os
import pandas as pd
import numpy as np
from ast import literal_eval
import pickle

from sklearn import linear_model
from scipy.special import softmax

def scale_to_sum_of_one(data):
    return data / data.sum()

with open('strategies.pkl', 'rb') as file:
    strategies = pickle.load(file)

full_data = None
for dataset in ['20News', 'NewsCategory', 'atis', 'facebook', 'liu', 'snips']:
# for dataset in ['NewsCategory', 'atis', 'facebook', 'liu', 'snips']:
    for MODEL in ['protonets', 'maml', 'finetuning', 'mistral', 'zephyr']:

        data = pd.read_csv(os.path.join('Data', dataset, f'{MODEL}.csv'))
        data['strategy'] = data.strategy.apply(literal_eval)

        mapping = {strategy: [] for strategy in strategies}
        for idx, row in data.iterrows():
            for strat in strategies:
                if strat in row.strategy:
                    mapping[strat].append(1)
                else:
                    mapping[strat].append(0)

        data = pd.concat([data, pd.DataFrame(mapping)], axis=1)
        if full_data is None:
            full_data = data
        else:
            full_data = pd.concat([full_data, data])
        y = data['diff_basic']
        X = data.drop(['strategy', 'acc', 'diff_random', 'diff_basic'], axis=1)


        clf = linear_model.Lasso(alpha=0.001, fit_intercept=True, positive=True)

        clf.fit(X, y)
        # print(clf.predict(X[0:1]))
        print(scale_to_sum_of_one(clf.coef_))
        for coef in scale_to_sum_of_one(clf.coef_):
            print(coef)

y = full_data['diff_basic']
X = full_data.drop(['strategy', 'acc', 'diff_random', 'diff_basic'], axis=1)

    # print(data)

clf = linear_model.Lasso(alpha=0.001, fit_intercept=True, positive=True)

clf.fit(X, y)
print(scale_to_sum_of_one(clf.coef_))
for coef in scale_to_sum_of_one(clf.coef_):
    print(coef)