# =============================================================================
#          File: get_dataset.py
#        Author: Andre Brener
#       Created: 14 Mar 2017
# Last Modified: 13 May 2017
#   Description: description
# =============================================================================
import os

import pandas as pd


def categorize_features(df, top_quant, mh_quant, ml_quant, low_quant):
    feature_cols = [col for col in df.columns if col.startswith('feature')]
    for col in feature_cols:
        max_value = df[col].max()
        top_cat = df[col].quantile(top_quant)
        mh_cat = df[col].quantile(mh_quant)
        ml_cat = df[col].quantile(ml_quant)
        low_cat = df[col].quantile(low_quant)
        min_value = df[col].min()
        bins = [min_value, low_cat, ml_cat, mh_cat, top_cat, max_value + 1]
        group_names = ['low', 'ml', 'm', 'mh', 'top']
        col_name = '{}_cat'.format(col)
        df[col_name] = pd.cut(
            df[col], bins, labels=group_names, include_lowest=True)
    print('Categorized :)')
    return df


def get_tables(df):
    categorical_cols = [col for col in df.columns if col.endswith('_cat')]
    train_table = df[categorical_cols]
    for col in categorical_cols:
        train_table = pd.concat(
            [
                train_table, pd.get_dummies(
                    train_table[col],
                    prefix=col,
                    prefix_sep='_',
                    dummy_na=False).astype(int)
            ],
            axis=1,
            join='inner')
        train_table.drop(col, axis=1, inplace=True)

    print('Tables Created :)')
    return train_table


def save_tables(data_path,
                df,
                top_quant,
                mh_quant,
                ml_quant,
                low_quant,
                train=True):

    df = categorize_features(df, top_quant, mh_quant, ml_quant, low_quant)
    X = get_tables(df)

    os.makedirs(data_path, exist_ok=True)

    x_path = '{}/tournament_table.csv'.format(data_path)

    if train:
        x_path = '{}/train_table.csv'.format(data_path)
        y = df['target'].to_frame()
        y.columns = ['target']
        y.to_csv('{}/target.csv'.format(data_path), index=False)
    else:
        ids = df['id'].to_frame()
        ids.columns = ['id']
        ids.to_csv('{}/ids.csv'.format(data_path), index=False)

    X.to_csv(x_path, index=False)


def dataset_main(file_path, data_path, tables_list, top_quant, mh_quant,
                 ml_quant, low_quant):
    for name, df, train in tables_list:
        print('\nTable: {}'.format(name))
        save_tables(data_path, df, top_quant, mh_quant, ml_quant, low_quant,
                    train)
    print('Files Saved')
