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
from sklearn.decomposition import PCA, KernelPCA

#params must be dict

def SVM(X, Y, ktype): 

	svmScores = []
	kf = KFold(n=len(Y), n_folds = nfolds)
	for train_index, test_index in kf:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]

		# SVC fit
		clf = svm.SVC(C=1.0, kernel= ktype)
		clf.fit(X_train, y_train)
		svmScores.append(clf.score(X_test, y_test))

	print "scores" , svmScores
		xx, yy = np.meshgrid(np.linspace(-10, 10, 500), np.linspace(-10, 10, 500))

		Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)

		plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=pl.cm.Blues_r)
		a = pl.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
		pl.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')

		colors = ['w' if i == 0 else 'k' for i in Y]
		plt.scatter(X[:,0], X[:,1], color = colors, alpha=1.0)

		# Plt that plot yoz. and make the xylim not shitty
		plt.xlim([np.min(X)-5,np.max(X)+5])
		plt.ylim([np.min(X)-5,np.max(X)+5])
		plt.show()

def logRegress(X,Y):

	scores = []
	for train_index, test_index in kf:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
		logModel = linear_model.LogisticRegression()
		logModel.fit(X_train,y_train)
		scores.append(logModel.score(X_test, y_test))
		
		print "Scores" , scores

		xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
		Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

		# Put the result into a color plot
		Z = Z.reshape(xx.shape)
		pl.figure(1, figsize=(4, 3))
		pl.pcolormesh(xx, yy, Z, cmap=pl.cm.Paired)

		# Plot also the training points
		pl.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=pl.cm.Paired)
		pl.xlabel('Sepal length')
		pl.ylabel('Sepal width')

		pl.xlim(xx.min(), xx.max())
		pl.ylim(yy.min(), yy.max())
		pl.xticks(())
		pl.yticks(())

		pl.show()


def pipeline (X, Y, Ktype, params, selectFeat, gridSearchOn = False,  logOn = False):

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
		PipeParams = dict('')
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
		grid_search = GridSearchCV(pipe, param_grid = PipeParams, scoring = 'accuracy', cv =5, verbose = 5)
		grid_search.fit(X,Y)
		print "best params: ", grid_search.best_params_
		print "best estimator: ", grid_search.best_estimator_
		print "scores : ", grid_search.grid_scores_

		#plotting gridsearch
		if modelName == 'RBF':
			pl.figure(figsize=(8, 6))
			pl.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95)
			pl.imshow(scores, interpolation='nearest', cmap=pl.cm.spectral)
			pl.xlabel('gamma')
			pl.ylabel('C')
			pl.colorbar()
			pl.xticks(np.arange(len(params['gamma'])), params['gamma'], rotation=45)
			pl.yticks(np.arange(len(params['C'])), params['C'])

		pl.show()
