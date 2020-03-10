# -*- coding: utf-8 -*-
# this file is used to pre-define basic params of models.

def get_defparams(model_name):
    """
    Get the pre-define params
    ----------------
    xgb; lasso; 
    wait: ANN; lgb
    """

    if model_name == 'xgb':
        return {'learning_rate': 0.1,
                'n_estimators': 60,
                'max_depth': 3,
                'min_child_weight': 1,
                'subsample': 1,
                'colsample_bytree': 1,
                'gamma':0 }
        
    elif model_name == 'lasso':
        return {'alpha': 20.0}

# add lgb:
    # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
    # https://github.com/XuesenYang/Several-Simple-Regression-Prediction-Methods/blob/master/main.py
    elif model_name == 'lgb':
        return {'objective': 'regression',
                'boosting_type': 'gdbt',
                'learning_rate': 0.1,
                'num_leaves': 3,
                'n_estimators': 500,
                'max_bin': 55
                }

# add ANN model to predict
    elif model_name == 'ANN':
        return {}