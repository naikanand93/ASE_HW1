# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 04:03:08 2017

@author: Anand
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 23:09:26 2017

@author: Anand
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import scipy.stats as sc
import urllib
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
#from sklearn.feature_selection import RFE,SelectPercentile
#from sklearn.metrics import r2_score
#from sklearn.model_selection import cross_val_score, GridSearchCV
import copy

###
#all the attributes identified as unrelated to the rent of an unoccupied apartment
#attributes related to the houseowner, attributes related to the personally owned house etc.
###
unrelated_attrs = [30 ,31 ,32 ,33 ,34 ,35 ,36 ,37 ,38 ,39 ,40 ,41 ,42 ,43 ,44 ,45 ,46 ,
                   47 ,48 ,49 ,50 ,51 ,52 ,53 ,54 ,55 ,56 ,57 ,58, 59 ,60,61,62,63,64,65, 66,91,93 ,94 ,95 ,
                   96 ,97 ,98 ,102,116,119 ,120 ,121 ,122 ,123 ,124 ,130 ,131 ,132 ,133 ,
                   134 ,135 ,140 ,141 ,142 ,143 ,144 ,145 ,146 ,147 ,148 ,149 ,150 ,151 ,
                   152 ,153 ,154 ,155 , 156 ,157 ,158 ,159 ,160 ,161 ,162 ,163 ,164 ,165 ,
                   166 ,167 ,168 ,169 ,170 ,171 ,172 ,173 ,174 ,175 ,187 ,188 ,189 ,190 ,
                   191 ,192 ,193 ,194 ,195 ,196 ]

###
#remaining features that will require one hot encoding in the end
###
feature_one_hot = [1,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,48,
                   50,52,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,
                   73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92]



def score_rent():
    """
    downloads the csv file and builds the model
    :return: R^2 score
    """
    response = urllib.request.urlretrieve('https://ndownloader.figshare.com/files/7586326')
    urlfile = response[0]

    df=pd.read_csv(urlfile, sep=',')
    target = df.values[:,91]


    ##removing all the rows for which response variable is missing before train/test split
    missing_response = np.where(target == 99999)
    df.drop(df.index[missing_response[0]],  inplace = True)
    target = np.delete(target, missing_response[0])
                   
    X_train, X_test, y_train, y_test = train_test_split(df.values, target, random_state=42)


    ##call to the function that imputes and transforms training data and then applies the similar transformations
    ##to the test data
    X_train, X_test = impute_preprocess_data(X_train, X_test)


    ##using standard scaler with mean = False as we have sparse data due to one hot encoding
    pipe = make_pipeline(preprocessing.StandardScaler(with_mean = False), Lasso(alpha=5.,random_state=41,max_iter=100000))#RFE(Lasso(alpha=5.,random_state=41), n_features_to_select=250))# 

##below is the commented code that I used to crossed to cross validate and predict regularizer for Lasso
#scoress = cross_val_score(pipe, X_train, y_train)
#print(scoress)
#param_grid = {'lasso__alpha':range(1,10,2)}
#grid = GridSearchCV(pipe,param_grid, cv = 10)
#grid.fit(X_train, y_train)
#print(grid.best_params_)
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    
    return score

#select = RFE(Lasso(alpha=10.,random_state=41), n_features_to_select=250)
#select.fit(X_train, y_train)


def predict_rent():
    """
        downloads the csv file and builds the model
        :return: test data, true test labels and predicted test labels
    """
    response = urllib.request.urlretrieve('https://ndownloader.figshare.com/files/7586326')
    urlfile = response[0]

    df=pd.read_csv(urlfile, sep=',')
    target = df.values[:,91]

    ##removing all the rows for which response variable is missing before train/test split
    missing_response = np.where(target == 99999)
    df.drop(df.index[missing_response[0]],  inplace = True)
    target = np.delete(target, missing_response[0])

    X_train, X_test, y_train, y_test = train_test_split(df.values, target, random_state=42)
    X_test_return = copy.deepcopy(X_test)

    ##call to the function that imputes and transforms training data and then applies the similar transformations
    ##to the test data
    X_train, X_test = impute_preprocess_data(X_train, X_test)

    pipe = make_pipeline(preprocessing.StandardScaler(with_mean = False), Lasso(alpha=5.,random_state=41,max_iter=100000))#RFE(Lasso(alpha=5.,random_state=41), n_features_to_select=250))# 
    pipe.fit(X_train, y_train)
    
    y_predicted = pipe.predict(X_test)
    
    return X_test_return, y_test, y_predicted

def impute_preprocess_data(X_train, X_test):
    df1 = pd.DataFrame(X_train)
    df2 = pd.DataFrame(X_test)

    ##List that will hold all the transformation that I apply on training data
    ##and will use it to apply the same on the test data
    transformations = []


    ##dropping all the unrelated attributes identified
    df1.drop(df1.columns[unrelated_attrs], axis=1, inplace = True)
    X_train= df1.values

    ##for the first few attributes replacing 9 with 0, to introduce sparseness
    for j in range(2,25):
        X_train[:,j][X_train[:,j] > 1] = 0

    #replacing with most_frequent for categorical attributes
    for j in range(25,30):
        mode = sc.mode(X_train[:,j])
        transformations.append(mode[0][0])
        X_train[:,j][X_train[:,j] == 8] = mode[0][0]

    #replacing with mean for the continuous attributes
    X_train[:,44][X_train[:,44] == 9999] = 473.0
    X_train[:,45][X_train[:,45] == 9999] = 597.0
    X_train[:,47][X_train[:,47] == 9999] = 3200.0
    X_train[:,51][X_train[:,51] == 9999] = 7800.0
    X_train[:,53][X_train[:,53] == 9999] = 325.0

    #replacing with most_frequent for categorical attributes
    for j in range(55,73):
        mode = sc.mode(X_train[:,j])
        transformations.append(mode[0][0])
        X_train[:,j][X_train[:,j] == 8] = mode[0][0]
    
    mode = sc.mode(X_train[:,77])
    X_train[:,77][X_train[:,77] == 8] = mode[0][0]
    transformations.append(mode[0][0])
    
    X_train[:,80][X_train[:,80] == 7] = 1

    ##applying one hot encoding to all the categorical attributes
    ohe = preprocessing.OneHotEncoder(categorical_features=feature_one_hot)
    X_train = ohe.fit(X_train).transform(X_train)
    
    ##############################
    ## below we apply all the same transformations to the test data
    ## such that there is no information leak
    ##############################

    df2.drop(df2.columns[unrelated_attrs], axis=1, inplace = True)
    X_test= df2.values

    ##for the first few attributes replacing 9 with 0, as training data to introduce sparseness
    for j in range(2,25):
        X_test[:,j][X_test[:,j] > 1] = 0
    ##replacing with most_frequent found from training set
    for j in range(25,30):
        X_test[:,j][X_test[:,j] == 8] = transformations[j-25]
    ##replacing with mean found in training set
    X_test[:,44][X_test[:,44] == 9999] = 473.0
    X_test[:,45][X_test[:,45] == 9999] = 597.0
    X_test[:,47][X_test[:,47] == 9999] = 3200.0
    X_test[:,51][X_test[:,51] == 9999] = 7800.0
    X_test[:,53][X_test[:,53] == 9999] = 325.0
    ##replacing with most_frequent found from training set
    for j in range(55,73):
        X_test[:,j][X_test[:,j] == 8] = transformations[j-55+5]
    X_test[:,77][X_test[:,77] == 8] = transformations[j-55+5]
    X_test[:,80][X_test[:,80] == 7] = 1

    ##transforming one hot encoding fit on training data
    X_test = ohe.transform(X_test)

    ##function returns the test and train data
    return X_train, X_test
    
def main():
    print(score_rent())  # predict_rent()
if __name__ == '__main__':
    main()
