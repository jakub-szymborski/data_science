"""
# -*- coding: utf-8 -*-
Created on Fri May 19 13:59:00 2023

@author: Kuba
sample data set - various parameters of portugese Vinho Verde wines (red and white)
https://archive.ics.uci.edu/ml/datasets/wine%20quality

variables: 
1 - fixed acidity
2 - volatile acidity
3 - citric acid
4 - residual sugar
5 - chlorides
6 - free sulfur dioxide
7 - total sulfur dioxide
8 - density
9 - pH
10 - sulphates
11 - alcohol
Output variable (based on sensory data):
12 - quality (score between 0 and 10)

toDo: 
    change dependant variable to be 'response' not 'alcohol' 
"""

# imports 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import sklearn.linear_model
import sklearn.preprocessing

# loading data 
wine = pd.read_csv("winequality-all.csv", comment="#")

# quick look at data 
wine.head()
wine.info()
wine.dtypes

# saving wine color as category 
wine.color = wine.color.astype("category")

#-----------------
# target of the analysis -> determining alcohol content based on parameters 

# checking number of red / white wine within the set 
wine.color.value_counts()

# separatation of dataset into red and white subsets 
white = wine[wine.color == 'white']
red = wine[wine.color == 'red']

# dropping color column, since no longer needed
white = white.drop('color', axis=1)
red = red.drop('color', axis=1)

# Data exploration 

# histograms 
fig, axis = plt.subplots(3, 4)

#create histogram for each column in DataFrame
white.hist(ax=axis)
plt.show()

# look for outliers, remove them, per below: 
# https://www.kaggle.com/code/semavasyliev/outliers-in-the-wine-quality-dataset






# dividing white subset into dependent (response) and independent variables  
# we're assuming that alcohol conent is dependant on all other parameters -> that might not be true 
X = white.loc[:,white.columns !='response']   # independent variables
y = white.loc[:,'response']                   # dependent (response) variable 
  
# division into train/ test sets 
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, train_size = 0.8)

# Pearson correlation 
corr_p = white.corr('pearson')
corr_p_half = corr_p.where(np.tril(np.ones(corr_p.shape),k=-1).astype(bool)) # half of correlation matrix - it's symmetrical so half is enough 

# list of correlations above abs(0.5)
corr_p_tri = corr_p.where(np.triu(np.ones(corr_p.shape, dtype=bool), k=1)).stack().sort_values()
corr_p_tri[abs(corr_p_tri)>0.5]

# correlation matrix plot  
with sns.axes_style('white'):
    f,ax = plt.subplots(figsize=(15,10))
    ax= sns.heatmap(corr_p_half,vmin=-1, vmax=1, cmap='RdBu', linewidth =.1, annot = True)
    plt.rc('xtick', labelsize=9) 
    plt.rc('ytick', labelsize=9)
    ax.tick_params(axis = 'x', labelrotation = 45)
    ax.xaxis.tick_top()    

# 2 visualisations showing correlations
    #  seaborn pairplot
if False:   # carefull, takes some time since there are many variables 
    sns.pairplot(white)
    plt.show()
    
        # pandas scatter matrix:
    scatter_matrix(white, alpha = 0.2, figsize = (6, 6), diagonal = 'hist')

# ------------- Machine Learning -----------------------------------
#linear regression models

# useful functions 
def tests(y_testing,y_predictions, model):
    # function comparing y_tested and y_predicted and calculating 4 metrics 
    R2  = round(sklearn.metrics.r2_score(y_testing,y_predictions),3)  # wsp. determinacji R2, idealna wartosc R2 = 1
    MSE = round(sklearn.metrics.mean_squared_error(y_testing,y_predictions),3) #MSE  -- im mniejszy tym lepszy 
    MAE = round(sklearn.metrics.mean_absolute_error(y_testing,y_predictions),3) #MAE   -- im mniejszy tym lepszy
    MedAe = round(sklearn.metrics.median_absolute_error(y_testing,y_predictions),3) #MedAE -- im mniejszy tym lepszy
    
    wyniki = pd.DataFrame.from_dict({'R2':R2, 'MSE':MSE, 'MAE':MAE, 'MedAe':MedAe}, orient = 'index', columns = [model] )
    return wyniki

def normalize(var):
    # function normalizing given variable into 0-1 range
    return (var - min(var))/(max(var) - min(var))

def fit_regression(X_train, X_test, y_train, y_test):
    # function for training and testing model, returns metrics for train and test datasets
    r = sklearn.linear_model.LinearRegression()
    r.fit(X_train, y_train)
    y_train_pred = r.predict(X_train)
    y_test_pred = r.predict(X_test)
    r2 = sklearn.metrics.r2_score
    mse = sklearn.metrics.mean_squared_error
    mae = sklearn.metrics.mean_absolute_error
    return {
        "r_score_train": round(r2(y_train, y_train_pred),3),
        "r_score_test": round(r2(y_test, y_test_pred),3),
        "MSE_train": round(mse(y_train, y_train_pred),3),
        "MSE_test": round(mse(y_test, y_test_pred),3),
        "MAE_train": round(mae(y_train, y_train_pred),3),
        "MAE_test": round(mae(y_test, y_test_pred),3) } 


# data normalization
white_normalne = white.copy()
colsToNormalize = white_normalne.columns     # all columns
white_normalne[colsToNormalize]   = white_normalne[colsToNormalize].apply(normalize)

X_norm = white_normalne.loc[:,white.columns !='alcohol']   # independent variables
y_norm = white_normalne.loc[:,'alcohol']                   # dependent variables 
X_train_n, X_test_n, y_train_n, y_test_n = sklearn.model_selection.train_test_split(X_norm,y_norm, train_size = 0.8)

# base model 
params = ['Base model']
res = [fit_regression(X_train, X_test,y_train,y_test)]
results = pd.DataFrame(res, index = params)

# base model, but data normalised
params.append('Normalised')
res.append(fit_regression(X_train_n,X_test_n,y_train_n,y_test_n))
results = pd.DataFrame(res, index = params)

# polynominal model
params.append('Polynominal')
poly = sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=True)
X_poly_train = poly.fit_transform(X_train_n)
X_poly_test = poly.fit_transform(X_test_n)
 
res.append(fit_regression(X_poly_train, X_poly_test, y_train_n, y_test_n))
results = pd.DataFrame(res, index = params)

# polynominal model, no bias to reduce parameters 
params.append('Polynominal no bias')
poly2 = sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)
X_poly_train2 = poly2.fit_transform(X_train_n)
X_poly_test2 = poly2.fit_transform(X_test_n)
 
res.append(fit_regression(X_poly_train2, X_poly_test2, y_train_n, y_test_n))
results = pd.DataFrame(res, index = params)





# version 1
"""
    alco = sklearn.linear_model.LinearRegression()
    alco.fit(x_train,y_train)
    
    # przygotowanie tabelki ze wszystkimi wspoczynnikami i zmiennymi
    alco_coeffs = pd.DataFrame(zip(x.columns, alco.coef_))
    alco_coeffs.loc[len(alco_coeffs)] = ['intecept', alco.intercept_]
    alco_coeffs.columns = ['variable', 'coefficent']
    
    # testowanie regresji liniowej / ocena dokladnosci 
    y_pred = alco.predict(x_test)
    wyniki_alco = tests(y_test,y_pred, 'bazowy')
    
    # nowy model - nowe uczenie
    alco_norm = sklearn.linear_model.LinearRegression()
    alco_norm.fit(x_train,y_train)
    
    # przygotowanie tabelki ze wszystkimi wspoczynnikami i zmiennymi
    alco_norm_coeffs = pd.DataFrame(zip(x.columns, alco_norm.coef_))
    alco_norm_coeffs.loc[len(alco_norm_coeffs)] = ['intecept', alco_norm.intercept_]
    alco_norm_coeffs.columns = ['variable', 'coefficent']
    
    # = nowy test 
    y_pred = alco.predict(x_test)
    
    wyniki_alco = wyniki_alco.join(tests(y_test,y_pred, 'znormalizowany'))

    # ------------ model wielomianowy ------------------------
    #korzystamy z funkcji PolynomialFeatures ze stopniem 2,
    #aby wygenerować nowe cechy, które są iloczynem cech bazowych,
    #np. [x1,x2,x3] -> [x1, x2, x3, x1^2, x1x2, x1x3, x2^2, x2x3, x3^2]
    
    #budujemy model wielomianowy przekształcając zbiór treningowy predyktorów X_train
    #oraz zbiór testowy predyktorów X_test
    wielomian2 = sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=True)
    x2_train = wielomian2.fit_transform(x_train)
    x2_test = wielomian2.fit_transform(x_test)
    
    x2_train.shape
    
    alco_x2 = sklearn.linear_model.LinearRegression()
    alco_x2.fit(x2_train,y_train)
    
    y_pred = alco_x2.predict(x2_test) 
    wyniki_alco = wyniki_alco.join(tests(y_test,y_pred, 'wielomian 2'))
    
    x2_train = pd.DataFrame(x2_train)
    alco_poly_coeffs = pd.DataFrame(zip(x2_train.columns, alco_x2.coef_))
    alco_poly_coeffs.loc[len(alco_poly_coeffs)] = ['intecept', alco_x2.intercept_]
    alco_poly_coeffs.columns = ['variable', 'coefficent']
"""