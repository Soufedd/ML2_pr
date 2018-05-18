import numpy as np
import functools as ft

from GPy.models import GPRegression, SparseGPRegression
from GPy.kern import RBF, Brownian, Linear

from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import RobustScaler, StandardScaler


from keras.models import Sequential
from keras.layers import Dense



class GPRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
	def __init__(self, max_sample = 500, kernel = None,  verbose = True, normalizer = None):
		if kernel is None:
			kernel = Brownian
		self.kernel = kernel
		self.models = []
		self.verbose = verbose
		self.normalizer = normalizer
		self.max_sample = max_sample

	def fit(self, X_train, Y_train):
		num_samples = len(X_train)
		if num_samples > self.max_sample:
			print('num_samples > {}, splitting training; {} steps'.format(self.max_sample,max(10,num_samples//self.max_sample)))
			i=0
			while i < max(10,num_samples//self.max_sample):
				#if self.verbose:
				print('step {}/{}'.format(i+1,max(10,num_samples//self.max_sample)))
				train_idx = np.random.choice(num_samples,self.max_sample,replace=False)
				X = X_train[train_idx]
				Y = Y_train[train_idx]
				kernels = []
				for dim in range(X_train.shape[1]):
					kernels.append(self.kernel(1, active_dims=[dim]))
					kern_multi = ft.reduce(lambda a,b: a+b, kernels)
				try:
					gpr = GPRegression(X, Y, kernel = kern_multi, normalizer = self.normalizer)
					gpr.optimize(messages=self.verbose,max_iters=200)
					self.models.append(gpr)
					i += 1
				except (np.linalg.LinAlgError):
					continue
				
		else:
			Repeat = True
			print('num_samples <= 300, single training')
			while Repeat:
				kernels = []
				for dim in range(X_train.shape[1]):
					kernels.append(self.kernel(1, active_dims=[dim]))
					kern_multi = ft.reduce(lambda a,b: a+b, kernels)
				try:
					gpr = GPRegression(X_train, Y_train, kernel = kern_multi, normalizer = self.normalizer)
					gpr.optimize(messages=self.verbose,max_iters=200)
					self.models.append(gpr)
					Repeat = False
				except (np.linalg.LinAlgError):
					continue

		return self


	def predict(self, X_test):
		mus = []
		vars = []
		for m in self.models:
		    mu,var = m.predict(X_test, full_cov=False)
		    mus.append(mu)
		    vars.append(var)
		w = np.array(vars)**(-2)/np.sum(np.array(vars)**(-2),axis = 0)
		var = 1/(np.sum(w/(np.array(vars)),axis=0))
		mu = var*(np.sum(np.array(w*mus)/np.array(vars),axis=0))
		return mu.reshape(1,-1)[0]





#Stack-Average Models Class
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
	def __init__(self, base_models, meta_model, n_folds=5):
		self.base_models = base_models
		self.meta_model = meta_model
		self.n_folds = n_folds

	# We again fit the data on clones of the original models
	def fit(self, X, y):
		self.base_models_ = [list() for x in self.base_models]
		self.meta_model_ = clone(self.meta_model)
		kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

		# Train cloned base models then create out-of-fold predictions
		# that are needed to train the cloned meta-model
		out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
		for i, model in enumerate(self.base_models):
			print('\nmodel {}/{}'.format(i+1,len(self.base_models)))
			j = 0
			splits = list(kfold.split(X,y))
			for train_index, holdout_index in splits:
				print('step {}/{}'.format(j+1,len(splits)))
				instance = clone(model)
				self.base_models_[i].append(instance)
				instance.fit(X[train_index], y[train_index])
				y_pred = instance.predict(X[holdout_index])
				out_of_fold_predictions[holdout_index, i] = y_pred
				j+=1

		# Now train the cloned  meta-model using the out-of-fold predictions as new feature
		self.meta_model_.fit(out_of_fold_predictions, y)
		return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
	def predict(self, X):
		meta_features = np.column_stack([
		np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
			for base_models in self.base_models_ ])
		return self.meta_model_.predict(meta_features)

class EnsembleRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
	def __init__(self, models, weights = None):
		if weights is None:
			weights = np.ones((len(models)))/len(models)
		assert(len(models)==len(weights))
		self.models = models
		self.weights = weights

	def fit(self, X, y):
		for model in self.models:
			model.fit(X,y)
		return self

	def predict(self, X):
		return np.array([self.weights[i]*model.predict(X) for i, model in enumerate(self.models)]).sum(axis = 0)





class PartScaler(TransformerMixin): 
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y):
        self.scaler.fit(X[:, :4], y)
        return self

    def transform(self, X):
        X_head = self.scaler.transform(X[:, :4])
        return np.concatenate([X_head, X[:, 4:]], axis=1)
