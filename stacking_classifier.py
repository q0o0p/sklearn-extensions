# -*- coding: utf-8 -*-
"""
@author: Anna Klepova (inspired by Faron)
"""

import operator
import numpy as np
from sklearn import cross_validation
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import LabelEncoder


class StackingClassifier:

    def __init__(self, **parameters):

        self.set_fields_by_default()
        self.set_params(**parameters)


    def set_fields_by_default(self):

        self.random_state = 0
        self.use_only_primary_clfs_results = True
        self.primary_is_proba = False
        self.label_encoder = LabelEncoder()
        self.classes_ = None


    def train_subsets(self, clf, X, y):

        nfolds = 5
        intermediate_answers = np.zeros((X.shape[0],))
        skf = cross_validation.StratifiedKFold(y,
                                               n_folds = nfolds,
                                               shuffle = True,
                                               random_state = self.random_state)
        clfs = []
        for i, (train_index, test_index) in enumerate(skf):
            x_tr = X[train_index]
            y_tr = y[train_index]
            x_te = X[test_index]
            clf.fit(x_tr, y_tr)
            clfs.append(clf)
            if self.primary_is_proba:
                intermediate_answers[test_index] = map(operator.itemgetter(1),
                                                       clf.predict_proba(x_te))
            else:
                intermediate_answers[test_index] = clf.predict(x_te)

        return intermediate_answers.reshape(-1, 1), tuple(clfs)


    def fit(self, X, y):

        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        intermediate_answers = dict()
        for i in range(len(self.primary_clfs)):
            (intermediate_answers[self.primary_clfs[i]['name']],
             self.primary_clfs[i]['clfs']) = self.train_subsets(self.primary_clfs[i]['clf'],
                                                                X,
                                                                y_encoded)
        features = [csr_matrix(intermediate_answers[self.primary_clfs[i]['name']]) for i in range(len(self.primary_clfs))]
        if not self.use_only_primary_clfs_results:
            features = [X] + features
        X_intermediate = hstack(features)
        self.clf.fit(X_intermediate, y_encoded)
        return self


    def build_matrix_for_prediction(self, X):

        nfolds = 5
        intermediate_answers = dict()
        for primary_clf in self.primary_clfs:
            intermediate_answers[primary_clf['name']] = np.zeros((X.shape[0],))
            intermediate_answers_skf = np.empty((nfolds, X.shape[0]))
            for i in range(nfolds):
                if self.primary_is_proba:
                    intermediate_answers_skf[i, :] = map(operator.itemgetter(1),
                                                         primary_clf['clfs'][i].predict_proba(X))
                else:
                    intermediate_answers_skf[i, :] = primary_clf['clfs'][i].predict(X)
            intermediate_answers[primary_clf['name']][:] = intermediate_answers_skf.mean(axis=0)
        features = [csr_matrix(intermediate_answers[primary_clf['name']].reshape(-1, 1)) for primary_clf in self.primary_clfs]
        if not self.use_only_primary_clfs_results:
            features = [X] + features
        X_intermediate = hstack(features)
        return X_intermediate
        
        
    def predict(self, X):
        
        X_intermediate = self.build_matrix_for_prediction(X)
        predicted = self.clf.predict(X_intermediate)
        return self.label_encoder.inverse_transform(predicted)


    def predict_proba(self, X):
        
        X_intermediate = self.build_matrix_for_prediction(X)
        predicted = self.clf.predict_proba(X_intermediate)
        return predicted


    def get_params(self, deep = True):
        
        params = {'primary_clfs': self.primary_clfs,
                  'clf': self.clf,
                  'use_only_primary_clfs_results': self.use_only_primary_clfs_results,
                  'primary_is_proba': self.primary_is_proba,
                  'label_encoder': self.label_encoder,
                  'classes_': self.classes_}
        for i in range(len(self.primary_clfs)):
            primary_clf_params = self.primary_clfs[i]['clf'].get_params()
            for k in primary_clf_params:
                params['primary__{}__{}'.format(self.primary_clfs[i]['name'], k)] = primary_clf_params[k]
        clf_params = self.clf.get_params()
        for k in clf_params:
            params['clf__{}'.format(k)] = clf_params[k]
        return params
    
    
    def set_params(self, **parameters):
        
        for param, value in parameters.items():
            if param == 'primary_clfs':
                if value is not None:
                    assert type(value) is list
                    self.primary_clfs = value
            if param == 'clf':
                if value is not None:
                    self.clf = value
            if param == 'random_state':
                if value is not None:
                    self.random_state = value
            if param == 'use_only_primary_clfs_results':
                if value is not None:
                    self.use_only_primary_clfs_results = value
            if param == 'primary_is_proba':
                if value is not None:
                    self.primary_is_proba = value
            if param == 'label_encoder':
                if value is not None:
                    self.label_encoder = value
            if param == 'classes_':
                if value is not None:
                    self.classes_ = value
        primary_clf_indices = dict()
        for i in range(len(self.primary_clfs)):
            primary_clf_indices[self.primary_clfs[i]['name']] = i
        primary_clf_new_params = dict()
        clf_new_params = dict()
        for param, value in parameters.items():
            splitted = param.split('__')
            if len(splitted) > 2:
                if splitted[0] == 'primary' and splitted[1] in primary_clf_indices:
                    if splitted[1] not in primary_clf_new_params:
                        primary_clf_new_params[splitted[1]] = dict()
                    primary_clf_new_params[splitted[1]]['__'.join(splitted[2:])] = value
                elif splitted[0] == 'clf':
                    clf_new_params['__'.join(splitted[1:])] = value
                else:
                    raise 'Error in StackingClassifier.set_params: wrong parameter name: {}\n'.format(param)
        for clf_name in primary_clf_new_params:
            self.primary_clfs[primary_clf_indices[clf_name]]['clf'].set_params(**(primary_clf_new_params[clf_name]))
        self.clf.set_params(**clf_new_params)
        return self


