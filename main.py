# =============================================================================
#          File: main.py
#        Author: Andre Brener
#       Created: 15 Mar 2017
# Last Modified: 13 May 2017
#   Description: description
# =============================================================================
import pandas as pd

from get_model import model_main
from get_dataset import dataset_main
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


def main(top_quant, mh_quant, ml_quant, low_quant, models):

    file_path = 'numerai_datasets'
    data_path = '{}/cats_{}_{}_{}_{}'.format(file_path, top_quant, mh_quant,
                                             ml_quant, low_quant)
    train_df = pd.read_csv('{}/numerai_training_data.csv'.format(file_path))
    tourn_df = pd.read_csv('{}/numerai_tournament_data.csv'.format(file_path))
    tables_list = [('train', train_df, True), ('tournament', tourn_df, False)]
    dataset_main(file_path, data_path, tables_list, top_quant, mh_quant,
                 ml_quant, low_quant)

    model_main(data_path, top_quant, mh_quant, ml_quant, low_quant, models)


forest_parameters = {
    'n_estimators': [250, 300],
    'max_features': ["auto", 20, 30],
    "bootstrap": [True, False],
    "min_samples_leaf": [1, 3],
    'max_depth': [5, 10]
}
SVR_parameters = {'C': [0.5, 0.7, 1.0]}
grad_parameters = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    "max_features": [None, 5],
    "max_leaf_nodes": [None, 5],
}

models = {
    'RandomForestRegressor': (RandomForestRegressor(), forest_parameters),
    'GradientBoostingRegressor':
    (GradientBoostingRegressor(), grad_parameters),
    'SVR': (SVR(), SVR_parameters)
}

categories = [[0.8, 0.6, 0.4, 0.2], [0.9, 0.7, 0.3, 0.1]]

for cat in categories:
    print(cat)
    top_quant, mh_quant, ml_quant, low_quant = cat

    main(top_quant, mh_quant, ml_quant, low_quant, models)
