import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA

#params must be dict


def pipeline (X, Y, Ktype, estimator, params, selectFeat, gridSearchOn = False,  logOn = False):

	modelName = ' ' 
	pipeScore = []
	
	#selecting features to fit into model with fregression. SelectFeat (boolean, int)
	if selectFeat[0]:
		features = SelectKBest(f_regression, k = selectFeat[1])
	
	#defining svc model and initializing for pipeline 
	if "Ktype" != 'linear':
		SVM = svm.SVC(kernel = Ktype)
		model = ('svm', SVM)
		pca = KernelPCA()
		modelName = Ktype
	else: 
		SVM = svm.LinearSVC()
		model = ('svm', SVM)
		pca = KernelPCA()
		modelName = 'Linear SVC '

	#log
	if logOn:
		logModel = linear_model.LogisticRegression()
		model = ('log' , logModel)
		pca = decomposition.PCA()
		modelName = 'Log Regression '

	#Pipeline creation: model = tuple of model name and model object
	if selectFeat[0]:
		pipe = Pipeline(steps =[('pca', pca), ('features', features), model] )
	else:
		pipe = Pipeline(steps=[('pca', pca), model])

	kf = KFold(n=len(Y), n_folds = 5)

	for train_index, test_index in kf:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
	#fit data
		pipe.fit(X_train,y_train)
		dfunct = pipe.decsion_function(X_train)
		pipeParam = pipe.get_params(X_train)
		pipeScore = pipe.score(X_test, y_test)
		score.append(pipeScore)
	
	print modelName, "scores for 5 tests", pipeScore

	# Run gridsearch
	if gridSearchOn:
		grid_search = GridSearchCV(pipe, param_grid = params, scoring = 'accuracy', cv =5, verbose = 5)
		grid_search.fit(X,Y)
		print "best params", grid_search.best_params_
