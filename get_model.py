# =============================================================================
#          File: get_model.py
#        Author: Andre Brener
#       Created: 15 Mar 2017
# Last Modified: 16 Mar 2017
#   Description: description
# =============================================================================
import pandas as pd

from sklearn.metrics import log_loss, mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split


def get_files(file_path, top_quant, mh_quant, ml_quant, low_quant):
    X = pd.read_csv('{}/train_table.csv'.format(file_path))
    y = pd.read_csv('{}/target.csv'.format(file_path))
    X_final = pd.read_csv('{}/tournament_table.csv'.format(file_path))
    ids = pd.read_csv('{}/ids.csv'.format(file_path))
    return X, y, X_final, ids


def pick_best_model_parameters(model, parameters, X_train, y_train):
    clf = GridSearchCV(model, parameters, cv=4, n_jobs=-1)
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    return clf.best_estimator_


def run_models(X, y, models):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    print('Starting modelization...')

    l = []

    for model in models:
        print('\nTraining Model', model)
        (clf, parameters) = models[model]
        l.append((model, pick_best_model_parameters(clf, parameters, X_train,
                                                    y_train)))

    return l, X_test, y_test


def get_results(l, X_test, y_test, X_final):
    final_preds = {}
    model_metrics = {}
    for model, clf in l:
        y_pred = clf.predict(X_test)
        msq = mean_squared_error(y_test, y_pred)
        logloss = log_loss(y_test, y_pred)

        print('\nModel: {}\n'.format(model))
        print('Mean Squared Error: {}'.format(msq))
        print('Log Loss: {}'.format(logloss))

        y_final_pred = clf.predict(X_final)
        y_final_pred = pd.DataFrame(y_final_pred, columns=['probability'])
        final_preds[model] = y_final_pred
        model_metrics[model] = (msq, logloss)

    model_metrics_df = pd.DataFrame(model_metrics)
    return final_preds, model_metrics_df


def predictions_csv(ids, final_preds, model_metrics_df, file_path):

    print('Predicting...')
    model_metrics_df.to_csv(
        '{}/model_metrics.csv'.format(file_path), index=False)
    for model in final_preds.keys():
        df = pd.merge(
            ids, final_preds[model], left_index=True, right_index=True)
        df.to_csv(
            '{}/{}_predictions.csv'.format(file_path, model), index=False)


def model_main(file_path, top_quant, mh_quant, ml_quant, low_quant, models):
    X, y, X_final, ids = get_files(file_path, top_quant, mh_quant, ml_quant,
                                   low_quant)
    l, X_test, y_test = run_models(X, y, models)
    final_preds, model_metrics_df = get_results(l, X_test, y_test, X_final)
    predictions_csv(ids, final_preds, model_metrics_df, file_path)
