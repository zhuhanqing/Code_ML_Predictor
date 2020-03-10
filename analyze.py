# -*- coding: utf-8 -*-
# This file is used to analyze the results and models.
# Now we can anylze the error; feature improtance; learning curve

import os
import pickle
import sys
import argparse

import numpy as np
import pandas as pd
import sklearn as sl
import sklearn.metrics as metrics
from sklearn.linear_model import Lasso
import xgboost as xgb
import lightgbm as lgb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

file_addr = '/data/Mia/1_cwy/HLS_Predictor/hls'

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_dir', type=str, 
                    default=file_addr + '/data/data_train.pkl', 
                    help='File of the training dataset. \
                    String. Default: ./data/data_train.pkl')

parser.add_argument('--test_data_dir', type=str, 
                    default=file_addr + '/data/data_test.pkl', 
                    help='File of the testing dataset. \
                    String. Default: ./data/data_test.pkl')

parser.add_argument('--model_save_dir', type=str, 
                    default=file_addr + '/saves/train/models_save.pkl', 
                    help='File of the pre-trained models. \
                    String. Default: ./save/train/models_save.pkl')

parser.add_argument('--param_save_dir', type=str, 
                    default=file_addr + '/saves/train/params.pkl', 
                    help = 'File of the pre-tuned params. \
                    String. Default: ./save/train/params_save.pkl')

parser.add_argument('--result_dir', type=str, 
                    default=file_addr + '/saves/test/results.pkl', 
                    help = 'File of the testing results. \
                    String. Default: ./save/test/results.pkl')

parser.add_argument('--save_result_dir', type=str, 
                    default=file_addr + '/saves/analysis/', 
                    help = 'Directory to save the analyzing results. \
                    String. Default: ./save/analysis/')

parser.add_argument('--save_pic_dir', type=str, 
                    default=file_addr + '/saves/analysis/pics/', 
                    help = 'Directory to save the analyzing pictures. \
                    String. Default: ./save/analysis/pics')

parser.add_argument('-f', '--func', type=str, default='sc', 
                    help='Select the analysis function. \
                    Value from "fi" or "feature_importance", \n \
                    "schls" or "score_hls", \
                    "sc" or "score" (default), \
                    "lc" or "learning_curve", \
                    "re" or "result_error", \
                    "rt" or "result_truth", \
                    "rp" or "result_predict", \
                    "rpd" or "result_predict_design".')

# function explaination
# fi - calculate feature importance
# sc - calculate the scores of model results
# schls - calculate the scores of HLS results
# lc - plot the learning curve
# re - show the result errors
# rt - show the result ground truth of the testing data
# rp - show the result prediction of the testing data
# rpd - show the result prediction grouped by the design ids


def load_data(file_name, silence=False):
    """
    Load data.
    """
    if not silence: print ('')
    if not silence: print ('Load data from: ', file_name)
    
    if not os.path.exists(file_name):
        sys.exit("Data file " + file_name + " does not exist!")
        
    with open(file_name, "rb") as f:
        print ()
        data = pickle.load(f, encoding='iso-8859-1')  
        
    # unpack the data
    X = data['x']
    Y = data['y']
    design_index = data['desii']
    feature_name = data['fname']
    target_name = data['tname']
    # use the mean and std to process raw data, then we need this to restre our data
    mean_features = data['fmean']
    mean_targets = data['tmean']
    std_features = data['fstd']
    std_targets = data['tstd']
    
    return X, Y, mean_features, mean_targets, std_features, std_targets, \
           feature_name, target_name, design_index
           
           
def load_test_data(FLAGS):
    """
    Load testing data.
    """
    return load_data(FLAGS.test_data_dir)


def load_train_data(FLAGS):
    """
    Load training data.
    """
    return load_data(FLAGS.train_data_dir)


def load_model_db(FLAGS):
    """
    Load model database.
    """
    # load models
    if not os.path.exists(FLAGS.model_save_dir):
        sys.exit("Model file " + FLAGS.model_save_dir + " does not exist!")
    else:
        return pickle.load(open(FLAGS.model_save_dir, "rb")) 


def load_result_db(FLAGS):
    """
    Load result database.
    """
    # load models
    if not os.path.exists(FLAGS.result_dir):
        sys.exit("Model file " + FLAGS.result_dir + " does not exist!")
    else:
        return pickle.load(open(FLAGS.result_dir, "rb"))
    
    
def load_param_db(FLAGS):
    """
    Load parameter database.
    """
    # load models
    if not os.path.exists(FLAGS.param_save_dir):
        sys.exit("Model file " + FLAGS.param_save_dir + " does not exist!")
    else:
        return pickle.load(open(FLAGS.param_save_dir, "rb")) 
           
           
def analyze_feature_importance(FLAGS):
    """
    Analyzing function: Analyze the feature importance.
    """
    # load models
    model_db = load_model_db(FLAGS)
    
    for key in model_db.keys():
        if key != 'lasso' and key != 'xgb':
            continue
        
        results = pd.DataFrame()

        for target in model_db[key].keys():
            # unpack the models
            model = model_db[key][target]['model']
            fnames = model_db[key][target]['fnames']
            
            # init column
            results[target] = pd.Series([None] * len(results), index=results.index, dtype=float)
            
            # which model
            ## lasso: use the model.coef_ to obtain the coefficent
            ## xgb  : use the feature weights to obtain the importance
#add lgb; ANN
            ## lgb  : use ???
            ## ANN  : use ???
            if key == 'lasso':
                feature_weights = abs(model.coef_)
            
            elif key == 'xgb':
                b = model.get_booster()
                feature_weights = [b.get_score(importance_type='weight').get(f, 0.) for f in b.feature_names]
                feature_weights = np.array(feature_weights, dtype=np.float32)
                feature_weights = feature_weights / feature_weights.sum()
            elif key == 'lgb':
                pass #wait to add
            
            elif key == 'ANN':
                pass

            else:
                continue
                
            for tt in range(len(feature_weights)):
                if fnames[tt] in results.index:
                    results[target][fnames[tt]] = feature_weights[tt]
                else:
                    results = results.append(pd.DataFrame([feature_weights[tt]], columns=[target], index=[fnames[tt]]))
            
        # save the results
        if not os.path.exists(FLAGS.save_result_dir):
            os.makedirs(FLAGS.save_result_dir)
        
        save_path = os.path.join(FLAGS.save_result_dir, 'feature_importance_' + key + '.csv')
        results.to_csv(save_path)
    
    # print
    print ('Succeed!') 
    print ('Analysis result is saved to', FLAGS.save_result_dir)
    
    # return 
    return results
    
    
def score_RAE(Y, Y_pre):
    """
    Calculate the score RAE.
    --------------
    Parameters:
        Y: The ground truth.
        Y_pre: The predicted values.
    """
    Y_pre = np.array(Y_pre)
    Y = np.array(Y)
    
    Y_mean = np.mean(Y)
    error = Y_pre - Y
    REA = np.mean(np.abs(error)) / (np.mean(np.abs(Y - Y_mean)) + np.finfo(float).eps)
    
    return REA


def score_RRSE(Y, Y_pre):
    """
    Calculate the score RRSE.
    --------------
    Parameters:
        Y: The ground truth.
        Y_pre: The predicted values.
    """
    return np.sqrt(1 - metrics.r2_score(Y, Y_pre))


def score_R2(Y, Y_pre):
    """
    Calculate the score R2.
    --------------
    Parameters:
        Y: The ground truth.
        Y_pre: The predicted values.
        
    """
    return metrics.r2_score(Y, Y_pre)


def analyze_scores(FLAGS):
    """
    Analyzing function: Calculate the scores of the result database.
    """
    # load results
    result_db = load_result_db(FLAGS)
    #print result_db 
    # traverse the results
    scores = {}
    for name in result_db.keys():
        scores[name] = {}
        for target in result_db[name].keys():
            # load result
            y_pre = result_db[name][target]['Pre']
            y_tru = result_db[name][target]['Truth']
            
            # metrics
            RAE = score_RAE(y_tru, y_pre)
            R2 = score_R2(y_tru, y_pre)
            RRSE = score_RRSE(y_tru, y_pre)
            
            # data to save
            scores[name][target] = {'RAE': RAE,
                                    'R2': R2,
                                    'RRSE': RRSE}
    
    # save
    if not os.path.exists(FLAGS.save_result_dir):
        os.makedirs(FLAGS.save_result_dir)

    # init
    # print ('scores.keys()[0]:', scores.keys()[0], '\nscores[scores.keys()[0]]:', scores[scores.keys()[0]], '\nscores[scores.keys()[0]].keys():', scores[scores.keys()[0]].keys()) 

    scores_RAE = pd.DataFrame(index=list(scores.keys()), columns=scores[list(scores.keys())[0]].keys())
    scores_R2 = pd.DataFrame(index=list(scores.keys()), columns=scores[list(scores.keys())[0]].keys())
    scores_RRSE = pd.DataFrame(index=list(scores.keys()), columns=scores[list(scores.keys())[0]].keys())
    
    for name in scores.keys():
        for target in scores[name].keys():
            scores_RAE[target][name] = scores[name][target]['RAE']
            scores_R2[target][name] = scores[name][target]['R2']
            scores_RRSE[target][name] = scores[name][target]['RRSE']
            
    scores_RAE.to_csv(os.path.join(FLAGS.save_result_dir, 'scores_RAE.csv'))
    scores_R2.to_csv(os.path.join(FLAGS.save_result_dir, 'scores_R2.csv'))
    scores_RRSE.to_csv(os.path.join(FLAGS.save_result_dir, 'scores_RRSE.csv'))
        
    print ('\nSucceed!') 
    print ('Analysis result is saved to', FLAGS.save_result_dir)
    
    # return
    return scores


def analyze_scores_hls(FLAGS):
    """
    Analyzing function: Calculate the scores of the HLS results.
    """
    # load testing data
    X, Y, mean_features, mean_targets, \
        std_features, std_targets, feature_names, target_names, \
        design_index = load_test_data(FLAGS)
    n = len(target_names)   
    # data to return
    scores = pd.DataFrame()
    
    # data to show
    scores_RAE = np.zeros([n], dtype=float)
    scores_R2 = np.zeros([n], dtype=float)
    scores_RRSE = np.zeros([n], dtype=float)
    
    # test model
    for ii in range(n):
        x = X[:, ii] * std_features[ii] + mean_features[ii]
        y = Y[:, ii] * std_targets[ii] + mean_targets[ii]
        
        # metrics
        scores_RAE[ii] = score_RAE(y, x)
        scores_R2[ii] =  sl.metrics.r2_score(y, x)
        scores_RRSE[ii] = np.sqrt(1 - sl.metrics.r2_score(y, x))
    
    # scores
    scores['Target'] = target_names[0: n]
    scores['RAE'] = scores_RAE
    scores['RRSE'] = scores_RRSE
    scores['1 - R2'] = 1 - scores_R2
    scores = scores.set_index('Target')
    
    # print scores
    print ('\nThe HLS scores are:')
    print (scores)
    
    # return 
    return scores


def analyze_result(FLAGS, value='error'):
    """
    Analyzing function: Calculate the error of the predicting results.
    """
    # load results
    result_db = load_result_db(FLAGS)
    
    # traverse the results

    for name in result_db.keys():
        for target in result_db[name].keys():
            # load result
            y_pre = result_db[name][target]['Pre']
            y_tru = result_db[name][target]['Truth']
            
            # calculate error and index
            if value == 'error':
                values = abs(y_pre - y_tru)
            elif value == 'truth':
                values = y_tru
            elif value == 'predict':
                values = y_pre
                
            index = [x for x in range(0, len(values))]
                
            # plot figure
            plt.figure()
            plt.bar(index, values)
            plt.title(name + ' for ' + target)
            plt.show()

            
            pic_dir = FLAGS.save_pic_dir
            pic_name =value + '_' + name + '_' + target + '_' + 'error.png'
            pic_path = os.path.join(pic_dir, pic_name)

            if not os.path.exists(pic_dir):
                    os.makedirs(pic_dir)
                
            plt.savefig(pic_path)
            


    print ('\nSucceed!') 


def analyze_learning_curve(FLAGS, eva_rule = 'RAE'):
    """
    Analyzing function: Draw the learning curve of the single models in the model database.
    """
    # load testing data
    X_test, Y_test, mean_features, mean_targets, \
        std_features, std_targets, feature_names, target_names, \
        design_index = load_test_data(FLAGS)
        
    # load training data
    X_train, Y_train, mean_features, mean_targets, \
        std_features, std_targets, feature_names, target_names, \
        design_index = load_train_data(FLAGS)
    
    # load params
    model_db = load_model_db(FLAGS)
    
    # traverse the models and params
    scores = {}
    for name in model_db.keys():
        if name not in ['xgb', 'lasso', 'lgb', 'ANN']: continue
        
        scores[name] = {}
        for ii in range(len(target_names)):
            target = target_names[ii]
            
            # get training info
            param = model_db[name][target]['model'].get_params()
            fsel = model_db[name][target]['fselect']
            
            # ? why need test different train number???
            # different training num
# issue1: 'float' object cannot be interpreted as an integer
# solution: https://blog.csdn.net/shuyueliang1/article/details/84996975
            # python 2: 1/200 = 0
            # python 3: 1/200 >> float number 
            # python 3: 1 // 200 >> int number
            train_nums = list(range(len(X_train) // 20 * 3, len(X_train), len(X_train) // 20))
            scores_test = []
            scores_train = []
            for num in train_nums:
                x_train = X_train[0: num, fsel]
                y_train = Y_train[0: num, ii]
                x_test = X_test[:, fsel]
                y_test = Y_test[:, ii]
                
                # choose model
# issue2: when use lasso, ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations.
# solution: sklearn.Lasso: https://www.jianshu.com/p/1177a0bcb306
# https://blog.csdn.net/qq_41185868/article/details/88209242
                # lasso model's params: we could set a smaller alpha(which may sacrifice the mdoel's accuracy)
                # or we add param setting to add the "max_iter=XXXX": 1000 or 10000
                if name == 'lasso':
                    np.random.seed(seed = 100)
                    model = Lasso(alpha=param['alpha'], max_iter=100000)
                    model.fit(x_train, y_train)
                    
                elif name == 'xgb':
                    np.random.seed(seed = 100)
                    model = xgb.XGBRegressor(objective ='reg:squarederror',
                                             learning_rate=param['learning_rate'],
                                             n_estimators=param['n_estimators'],
                                             max_depth=param['max_depth'],
                                             min_child_weight=param['min_child_weight'],
                                             subsample=param['subsample'],
                                             colsample_bytree=param['colsample_bytree'],
                                             #colsample_bytree=0.1,
                                             gamma=param['gamma'])
                    model.fit(x_train, y_train)
                elif name == 'lgb':
                    #model.fit(x_train, y_train)
                    pass
                
                elif name == 'ANN':
                    pass

                y_train_pre = model.predict(x_train)
                y_test_pre = model.predict(x_test)
                
                if eva_rule == 'RRSE':
                    scores_train.append(score_RRSE(y_train, y_train_pre))
                    scores_test.append(score_RRSE(y_test, y_test_pre))
                elif eva_rule == 'RAE':
                    scores_train.append(score_RAE(y_train, y_train_pre))
                    scores_test.append(score_RAE(y_test, y_test_pre))
                elif eva_rule == 'R2':
                    scores_train.append(score_R2(y_train, y_train_pre))
                    scores_test.append(score_R2(y_test, y_test_pre))
            
            if len(scores_train) > 0:
                # data to save
                scores[name][target] = {'nums': train_nums,
                                        'train': scores_train,
                                        'test': scores_test}
                
                # plot figure
                # plot the figure: RRSE
                plt.figure()
                plt.plot(scores[name][target]['nums'], scores[name][target]['train'])
                plt.plot(scores[name][target]['nums'], scores[name][target]['test'])
                plt.legend(['train', 'test'])
                plt.title(name + ' for ' + target)
                plt.show()
# issue3: python plt could not show the picture by matplotlib
# solution: store the picture to local dir, then open it at local mahcine
# https://blog.csdn.net/ydyang1126/article/details/77247654
                # obtain the picture name
                pic_dir = FLAGS.save_pic_dir
                pic_name = eva_rule + '_' + name + '_' + target + '_' + 'train_and_test_learning_curve.png'
                pic_path = os.path.join(pic_dir, pic_name)
# issue4: makedir problem: cannot create the directory
# solution: https://blog.csdn.net/ljl6158999/article/details/70807738
                # if we have 'A/B/C', then we create a son-directory in C
                    # use os.mkdir(A/B/C/D)
                # if we want to create D and E
                    # we need use os.makedirs(A/B/C/D/E) sine we donnot have D
                if not os.path.exists(pic_dir):
                    os.makedirs(pic_dir)
                
                plt.savefig(pic_path)
                
    
    # return
    return scores


if __name__ == '__main__':
    # parser
    FLAGS, unparsed = parser.parse_known_args()
    
    # choose function
    if FLAGS.func == 'feature_importance' or FLAGS.func == 'fi':
        results = analyze_feature_importance(FLAGS)
        
    elif FLAGS.func == 'sores' or FLAGS.func == 'sc':
        results = analyze_scores(FLAGS)
        
    elif FLAGS.func == 'sores_hls' or FLAGS.func == 'schls':
        results = analyze_scores_hls(FLAGS)
        
    elif FLAGS.func == 'learning_curve' or FLAGS.func == 'lc':
        results = analyze_learning_curve(FLAGS)
        
    elif FLAGS.func == 'result_error' or FLAGS.func == 're':
        results = analyze_result(FLAGS, value='error')
        
    elif FLAGS.func == 'result_predict' or FLAGS.func == 'rp':
        results = analyze_result(FLAGS, value='predict')
        
    elif FLAGS.func == 'result_truth' or FLAGS.func == 'rt':
        results = analyze_result(FLAGS, value='truth')
        
    elif FLAGS.func == 'result_truth_design' or FLAGS.func == 'rtd':
        results = analyze_result_design(FLAGS, value='truth')

# error picture
