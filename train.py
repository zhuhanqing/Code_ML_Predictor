# -*- coding: utf-8 -*-
# This file is used to train models and save the model into model database.
import os
import pickle
import sys
import argparse

import numpy as np
import xgboost as xgb
import lightgbm as lgb
import sklearn as sl
from sklearn.linear_model import Lasso

from params import get_defparams

#add local file address
fileAddr = '/data/Mia/1_cwy/HLS_Predictor/'

#add parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, \
                    default=fileAddr + '/data/data_train.pkl', 
                    help='Directory or file of the training dataset. \
                    String. Default: ./data/data_train.pkl')

parser.add_argument('--params_dir', type=str, \
                    default=fileAddr + '/saves/train/params.pkl',
                    help='Directory or file to load the parameters. \
                    String. Default: ./saves/train/params.pkl')

parser.add_argument('--params_save_dir', type=str, \
                    default=fileAddr + '/saves/train/params_save.pkl',
                    help='Directory or file to load the parameters. \
                    String. Default: ./saves/train/params_save.pkl')

parser.add_argument('--models_dir', type=str, \
                    default=fileAddr + '/saves/train/models.pkl', 
                    help='Directory or file to load the trained model. \
                    String. Default: ./saves/train/models.pkl')

parser.add_argument('--models_save_dir', type=str, \
                    default=fileAddr + '/saves/train/models_save.pkl',
                    help='Directory or file to load the trained model. \
                    String. Default: ./saves/train/models_save.pkl')

parser.add_argument('-d', '--disable_param_tuning', action='store_true',
                     help='Whether to disable parameters tuning or not. \
                     Boolean. Default: false')

parser.add_argument('--validation_ratio', type=float, default=0.25,
                     help='The ratio of the training data to do validation. \
                     Float. Default: 0.25')

parser.add_argument('-m', '--model_train', type=str, default='xgb',
                     help='The model to be trained. \
                     Empty means not training models. \
                     Value from "", "xgb"(default), "lasso"')

parser.add_argument('-s', '--model_fsel', type=str, default='lasso',
                     help='The model used to select features. \
                     Empty means not selecting features. \
                     Value from "", "xgb", "lasso"(default)') 


def load_data(FLAGS, silence=False):
    """
    Load training dataset.
    """
    if not silence: print ('')
    if not silence: print ('Load data from: ', FLAGS.data_dir)
    
    # check file exist
    if not os.path.exists(FLAGS.data_dir):
        sys.exit("Data file " + FLAGS.data_dir + " does not exist!")
    
    # load data
#Note!!! encoding format
    # byte would let the key be byte stream not the string you need
    with open(FLAGS.data_dir, "rb") as f:
        print ()
        data = pickle.load(f, encoding='iso-8859-1')
    
    # unpack the data
    ## store the data into corresponding area: X, Y, features, targets, design index
    X = data['x']
    Y = data['y']
    design_index = data['desii']
    feature_name = data['fname']
    target_name = data['tname']

    for ii in range(0, len(target_name)):
        print (target_name[ii], ' ')
    
    return X, Y, feature_name, target_name, design_index


def load_model_db(FLAGS, silence=False):
    """
    Load model database.
    """
    if not silence: print ('')
    if not silence: print ('Load model from: ', FLAGS.models_dir)
    
    if not os.path.exists(FLAGS.models_dir):
        return {}
        print ('no')
        
    with open(FLAGS.models_dir, "rb") as f:     
        model_db = pickle.load(f, encoding='iso-8859-1')    
    return model_db

def save_models(name, models, FLAGS, silence=False):
    """
    Save the model to the model database.
    --------------
    Parameters:
        name: The key of the model to be saved.
        models: The model to be saved.    
    """
    if models is None or len(models) == 0:
        return 
    
    # input file name
    file_dir, file_name = os.path.split(FLAGS.models_save_dir)
    if file_dir == '': file_dir = "/data/Mia/1_cwy/HLS_Predictor/hls/saves/train/"
    if file_name == '': file_name = 'models_save.pkl'
    file_path = os.path.join(file_dir, file_name)

    if not silence: print ('file path:', file_path)
    
    # create folder
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    
    # load model database
    model_db = load_model_db(FLAGS, silence=True)
    
    # check load model
    if model_db == {}:
        print ("Load model database unsuccessfuly")
    else:
        print ("Load model database successfuly")
        print ()
        print (model_db)

    print ('hi')
    print (models)
 
    # modify model database
    model_db[name] = models
    
#Note!!!
    # In Python 3.X, you have to specify the binary model, that is 'wb', 'rb'
    # Create files
    pickle.dump(model_db, open(file_path, "wb"))
        
    if not silence: print ('')
    if not silence: print ('Save Models to: ', file_path)

def load_param_db(FLAGS, silence=False):
    """
    Load model database.
    """
    if not silence: print ('')
    if not silence: print ('Load param from: ', FLAGS.params_dir)

    if not os.path.exists(FLAGS.params_dir):
        print ('no')
        return {}

    with open(FLAGS.params_dir, "rb") as f:
        param_db = pickle.load(f,encoding='iso-8859-1')

    return param_db
    
    
def save_params(name, params, FLAGS, silence=False):
    """
    Save the parameters to the parameter database.
    --------------
    Parameters:
        name: The key of the parameter to be saved.
        params: The parameter to be saved.    
    """
    if params is None or len(params) == 0:
        return 
    
    # input file name
    file_dir, file_name = os.path.split(FLAGS.params_save_dir) 
    if file_dir == '': file_dir = "/data/Mia/1_cwy/HLS_Predictor/hls/saves/train/"
    if file_name == '': file_name = 'params_save.pkl'
    file_path = os.path.join(file_dir, file_name)
    
    # create folder
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    
    # load model database
    param_db = load_param_db(FLAGS, silence=True)
    
    #check the load params
    if param_db == {}:
        print ("Load params database unsuccessfuly")
    else:
        print ("Load params database successfuly")
        print()
        print (param_db)

    print (params)
    
    param_db[name] = params
    
    # create file
    pickle.dump(param_db, open(file_path, "wb"))
        
    if not silence: print ('')
    if not silence: print ('Save Parameters to: ', file_path)
    
    
def train_models(X, Y, design_index, FLAGS, silence=False):
    """
    Train model.
    --------------
    Parameters:
        X: Feature dataset.
        Y: Target dataset. 
        design_index: Design id of the dataset. It's used to partition the validation dataset.
    --------------
    Return: 
        models: The trained model.
        params: The tuned parameters.
    """
    global Feature_Names, Target_Names
    
    if not silence: print ('')
    
    # data to return
    models = {}
    params = {}
  
    # load parameters
    if os.path.exists(FLAGS.params_dir):
        params_db = pickle.load(open(FLAGS.params_dir, 'rb'))
        if FLAGS.model_train in params_db.keys():
            params = params_db[FLAGS.model_train]
            if not silence: print ('we find the parameters when load parameters')

    for ii in range(0, len(Target_Names)):  
        target = Target_Names[ii]
        x = X
        y = Y[:, ii]
        
        #if not silence: print ('')
        if not silence: print ('For target -'), target, ('...')
        
        # select features
        if FLAGS.model_fsel != '':
            if not silence: print ('Selecting features by'), FLAGS.model_fsel, (' ...')
            
            feature_select = select_features(FLAGS.model_fsel, x, y)
        else:
            feature_select = np.ones(x.shape[1], dtype=bool)
        
        # load the pre-define parameters from params.py
        if target not in params.keys():
            param = get_defparams(FLAGS.model_train)
        else: 
            param = params[target]['param']

#Note!!!
        # here we tune the params for the models            
        # tune parameters
        if not FLAGS.disable_param_tuning:
            if not silence: print ('Tuning parameters for', FLAGS.model_train, '  ...')
            
            # tune param for LASSO
            if FLAGS.model_train == 'lasso':
#Note: adjust the alpha to control the hyperparameters
                param = tune_parameters('lasso', x[:, feature_select], y, design_index, 
                                        param, [{'alpha': t / 200.0} for t in range(0, 100, 1)],
                                        valid_ratio=FLAGS.validation_ratio)
            
            # tune param for XGBoost
            if FLAGS.model_train == 'xgb':
                # sequential tuning for 5 iterations
#Note: tune the params for xgb step by step
                # we need adjust these parameters to obtain a better one
                # refer to  https://blog.csdn.net/u014732537/article/details/80055227
                for tt in range(3): # 5
                    param = tune_parameters('xgb', x[:, feature_select], y, design_index, 
                                            param, [{'learning_rate': t / 50.0} for t in range(1, 40, 1)],
                                            valid_ratio=FLAGS.validation_ratio)
                    
                    param = tune_parameters('xgb', x[:, feature_select], y, design_index, 
                                            param, [{'n_estimators': t} for t in range(2, 100, 2)],
                                            valid_ratio=FLAGS.validation_ratio)
                    
                    param = tune_parameters('xgb', x[:, feature_select], y, design_index, 
                                            param, [{'max_depth': t} for t in range(2, 11, 1)],
                                            valid_ratio=FLAGS.validation_ratio)
                    
                    param = tune_parameters('xgb', x[:, feature_select], y, design_index, 
                                            param, [{'min_child_weight': t} for t in range(0, 51, 2)],
                                            valid_ratio=FLAGS.validation_ratio)
                    
                    param = tune_parameters('xgb', x[:, feature_select], y, design_index, 
                                            param, [{'subsample': t / 50.0} for t in range(10, 51, 1)],
                                            valid_ratio=FLAGS.validation_ratio)
                    
                    param = tune_parameters('xgb', x[:, feature_select], y, design_index, 
                                            param, [{'colsample_bytree': t / 50.0} for t in range(10, 51, 1)],
                                            valid_ratio=FLAGS.validation_ratio)
                    
                    param = tune_parameters('xgb', x[:, feature_select], y, design_index, 
                                            param, [{'gamma': t / 100.0} for t in range(0, 101, 2)],
                                            valid_ratio=FLAGS.validation_ratio)
                    
                    param['learning_rate'] = param['learning_rate'] / 2.0
                    param['n_estimators'] = param['n_estimators'] * 3
            
        # train model
        if not silence: print ('Training ...')
        if not silence: print ('Parameters for ' + FLAGS.model_train + ':', param)
            
        # train model - xgboost
        if FLAGS.model_train == 'xgb':
            np.random.seed(seed = 100)
#Issue: linear is now deprecated in favor of reg:linearerror.
#Solved: add the obj to solve the following issue   
            model = xgb.XGBRegressor(objective ='reg:squarederror',
                                     learning_rate=param['learning_rate'],
                                     n_estimators=param['n_estimators'],
                                     max_depth=param['max_depth'],
                                     min_child_weight=param['min_child_weight'],
                                     subsample=param['subsample'],
                                     colsample_bytree=param['colsample_bytree'],
                                     #colsample_bytree=0.1,
                                     gamma=param['gamma'])
            model.fit(x[:, feature_select], y)
        
#Issue: add lightgbm
#Unsolved
        elif FLAGS.model_train == 'lgb':
            np.random.seed(seed == 100)
            '''
            model = lgb.LGBMRegressor(learning_rate=param['learning_rate'],
                                     n_estimators=param['n_estimators'],
                                     max_depth=param['max_depth'],
                                     min_child_weight=param['min_child_weight'],
                                     subsample=param['subsample'],
                                     colsample_bytree=param['colsample_bytree'],
                                     #colsample_bytree=0.1,
                                     gamma=param['gamma'])
            '''

        # train model - lasso
        elif FLAGS.model_train == 'lasso':
            np.random.seed(seed = 100)
#Issue: need more iteration to fit the model
#Solved: add max_iter
            model = Lasso(alpha=param['alpha'], max_iter = 10000)
            model.fit(x[:, feature_select], y)

        elif FLAGS.model_train == 'ANN':
            pass
                
        # add to list to save
        params[target] = {'param': param }
        
        # add to list to return
        models[target] = {'model': model, 
                          'fselect': feature_select,
                          'fnames': np.array(Feature_Names)[feature_select].tolist()}
         
    # return

    return models, params


#Issue: try to assemble multi model
def assemble_models(X, Y, model_db, FLAGS, silence=False):
    """
    Assemble model. Assemble 2 or more models in the model database to be one model.
    --------------
    Parameters:
        X: Feature dataset.
        Y: Target dataset. 
        model_db: Model database.
    """
    global Feature_Names, Target_Names
    
    if not silence: print ('')
    
    # train model
    if not silence: print ('Assembling', FLAGS.model_assemble, ' ...')
        
    # data to return
    models = {}

    # wait

    # return
    return models


def select_features(strategy, X, Y, silence=False):
    """
    Select features.
    --------------
    Parameters:
        X: Feature dataset.
        Y: Target dataset. 
        strategy: Strategy used to select features.
            xgb - Select by XGBoost
            lasso - Select by LASSO
    --------------
    Return:
        feature_select: Feature selecting result. It's a boolean array. True means the feature is selected.
    """
    # fix the random seed
    np.random.seed(seed = 100)
    
    """
    Select features
        xgb: 0.03 based on the weights
        lasso: based on the model.coefficient
    https://blog.csdn.net/John_xyz/article/details/85165064?utm_source=distribute.pc_relevant.none-task
    """
    if strategy == 'xgb':
        model = xgb.XGBRegressor()
        model.fit(X, Y)
        
        b = model.get_booster()
        feature_weights = [b.get_score(importance_type='weight').get(f, 0.) for f in b.feature_names]
        feature_weights = np.array(feature_weights, dtype=np.float32)
        feature_select  = (feature_weights / feature_weights.sum()) > 0.03
        # refer to wisc paper; we have to adjust the importance threshold since sometimes more features input could be noise

    elif strategy == 'lasso':
        model = Lasso(alpha=0.01)
        model.fit(X, Y)
        feature_select = model.coef_ != 0

# add lightgbm strategy    
    elif strategy == 'lgb':
        model = lgb.LGBMRegressor()
        model.fit(X, Y)

        #? still have feature importance

    return feature_select


def tune_parameters(model_name, X, Y, design_index, def_param, tune_params, 
                    valid_ratio=0.25, silence=False):
    """
    Tune parameters.
    --------------
    Parameters:
        X: Feature dataset.
        Y: Target dataset. 
        model_name: The model to tune parameters for.
        design_index: Design id of the dataset. It's used to partition the validation dataset.
        def_param: The default parameters before tuning. It's a dict.
        tune_params: The parameters list where the best parameter to be chosen from.
        valid_ratio: Ratio of the validation data in all the training data.
    --------------
    Return:
        param: The tuned parameters. It's a dict.
    """
    # fix the random seed
    np.random.seed(seed = 100)
    
    # initial
    params = []
    scores = []
    
    # construct parameters
    if len(tune_params) == 0:
        params = [def_param]
    else:
        # [{'min_child_weight': t} for t in range(0, 51, 2)]
        # here you can see tune_params is a list with each element as a dict: name: value

        for tune_param in tune_params:
            param = dict(def_param)
            # let param become a complete params set and then change its value waiting for tune
            for key in tune_param.keys():
                # let tune_param's value to param and then add this param after params
                param[key] = tune_param[key]
                params.append(param)
    # check the params
    if not silence: print(params)

    # tune method
    # validation group
    ids = np.unique(design_index)
    vld_ids = []
    for ii in range(int(1 / valid_ratio)):
        start = int(len(ids) * ii * valid_ratio)
        end = int(len(ids) * (ii + 1) * valid_ratio)
        vld_ids.append(ids[start: end])
        
    # tune parameters
    for param in params:
        # get valid ids
        Y_pre = []
        Y_vld = []
        
        # cross validation
        for vld_id in vld_ids:
            # partition
            xt = X[[x not in vld_id for x in design_index]]
            yt = Y[[x not in vld_id for x in design_index]]
            xv = X[[x in vld_id for x in design_index]]
            yv = Y[[x in vld_id for x in design_index]]
            
            # init
            if model_name == 'lasso':
#Issue: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations.
#Solved: add the max_iter in the model's setting
                model = Lasso(alpha=param['alpha'], max_iter=10000)
                
            elif model_name == 'xgb':
                model = xgb.XGBRegressor(objective ='reg:squarederror',
                                         learning_rate=param['learning_rate'],
                                         n_estimators=param['n_estimators'],
                                         max_depth=param['max_depth'],
                                         min_child_weight=param['min_child_weight'],
                                         subsample=param['subsample'],
                                         colsample_bytree=param['colsample_bytree'],
                                         gamma=param['gamma'])
            else:
                break
                
            # train
            model.fit(xt, yt)
            
            # predict
            yp = model.predict(xv)
            
            # add to list
            Y_vld.extend(yv)
            Y_pre.extend(yp)
            
        # score
        score = sl.metrics.r2_score(Y_vld, Y_pre)
        
        # add to list
        scores.append(score)
        
        if not silence: print (model_name, ' -', param, 'R2 Score =', score)
    
    # get tuned parameters
    param = params[np.argmax(scores)]
    
    # return 
    return param
    
    
if __name__ == '__main__':
    global Feature_Names, Target_Names
    
    # parser
    FLAGS, unparsed = parser.parse_known_args()
    
    # print info
    print ("\n========== Start training models ==========\n")
    
    # load training data
    X_train, Y_train, Feature_Names, Target_Names, design_index = load_data(FLAGS)
    
    # when we do NOT hope to train assemble model
    if FLAGS.model_assemble == '':	
        if FLAGS.model_train != '':
            
            # train models
            models, params = train_models(X_train, Y_train, design_index, FLAGS)
        
            # save models
            print ("\n========== Save training models: ", FLAGS.model_train," ==========\n")
            save_models(FLAGS.model_train, models, FLAGS)
        
            # save params
            save_params(FLAGS.model_train, params, FLAGS)
            

    # when we hope to train assemble model
    else:
        pass
    
    print ("\n========== End ==========\n")
