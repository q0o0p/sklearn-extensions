# -*- coding: utf-8 -*-
"""
@author: Anna Klepova (inspired by Faron from Kaggle)
"""

import sys
import operator
import numpy as np
from copy import deepcopy
from sklearn import cross_validation
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import LabelEncoder


class StackingClassifier:

    def __init__(self, **parameters):

        self.set_fields_by_default()
        self.set_params(**parameters)


    def set_fields_by_default(self):

        self.version = '0.0.2'
        self.random_state = 0
        self.use_only_primary_clfs_results = True
        self.primary_is_proba = False
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        self.mode_ = None # BINARY or MULTI
        self.primary_answers_choosing_method = 'MEANS' # 'MEANS', 'BEST_CLF', 'RANDOM_CLF'


    def train_subsets(self, clf, X, y):

        nfolds = 5
        if self.mode_ == 'BINARY':
            intermediate_answers = np.zeros((X.shape[0],))
        else:
            assert self.mode_ == 'MULTI'
            intermediate_answers = np.zeros((X.shape[0], len(self.classes_)))
        skf = cross_validation.StratifiedKFold(y,
                                               n_folds = nfolds,
                                               shuffle = True,
                                               random_state = self.random_state)
        clfs = []
        for i, (train_index, test_index) in enumerate(skf):
            x_tr = X[train_index]
            y_tr = y[train_index]
            x_te = X[test_index]
            clf_copy = deepcopy(clf)
            clf_copy.fit(x_tr, y_tr)
            clfs.append(clf_copy)
            if self.primary_is_proba:
                if 'predict_proba' in dir(clf_copy):
                    probas = clf_copy.predict_proba(x_te)
                    if self.mode_ == 'BINARY':
                        intermediate_answers[test_index] = map(operator.itemgetter(1),
                                                               probas)
                    else:
                        assert self.mode_ == 'MULTI'
                        intermediate_answers[test_index] = probas
                elif 'decision_function' in dir(clf_copy):
                    scores = clf_copy.decision_function(x_te)
                    if self.mode_ == 'BINARY':
                        assert type(scores[0]) is np.float64
                    else:
                        assert self.mode_ == 'MULTI'
                        assert type(scores[0]) is np.ndarray
                    intermediate_answers[test_index] = scores
                else:
                    print >> sys.stderr, 'Error in StackingClassifier: parameter "primary_is_proba" is True, however one of primary classifiers has neither "predict_proba" method nor "decision_function". This classifier is "{}"\n'.format(type(clf_copy))
                    exit(1)
            else:
                intermediate_answers[test_index] = clf_copy.predict(x_te)

        if self.mode_ == 'BINARY':
            return intermediate_answers.reshape(-1, 1), tuple(clfs)
        else:
            assert self.mode_ == 'MULTI'
            return intermediate_answers, tuple(clfs)


    def fit(self, X, y):

        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        if len(self.classes_) < 2:
            print >> sys.stderr, 'Error in StackingClassifier: There are {} classes. There should be at least 2 classes\n'
            exit(1)
        elif len(self.classes_) == 2:
            self.mode_ = 'BINARY'
        else:
            self.mode_ = 'MULTI'
        intermediate_answers = dict()
        for i in range(len(self.primary_clfs)):
            (intermediate_answers[self.primary_clfs[i]['name']],
             self.primary_clfs[i]['clfs']) = self.train_subsets(self.primary_clfs[i]['clf'],
                                                                X,
                                                                y_encoded)
            #print "type(intermediate_answers[self.primary_clfs[i]['name']])", type(intermediate_answers[self.primary_clfs[i]['name']])
            #print "intermediate_answers[self.primary_clfs[i]['name']].shape", intermediate_answers[self.primary_clfs[i]['name']].shape
        if self.mode_ == 'BINARY':
            features = [csr_matrix(intermediate_answers[self.primary_clfs[i]['name']]) for i in range(len(self.primary_clfs))]
        else:
            assert self.mode_ == 'MULTI'
            # [clf1_cls1, clf1_cls2, clf1_cls3, clf2_cls1, clf2_cls2, clf2_cls3,]
            features = [csr_matrix(intermediate_answers[self.primary_clfs[i]['name']][:, class_idx].reshape(-1, 1)) for i in range(len(self.primary_clfs)) for class_idx in range(len(self.classes_))]
        if not self.use_only_primary_clfs_results:
            features = [X] + features
        #print 'len(features)', len(features)
        #print 'features[0].shape in fit', features[0].shape
        X_intermediate = hstack(features)
        #print 'X_intermediate.shape in fit', X_intermediate.shape
        self.clf.fit(X_intermediate, y_encoded)
        #print 'fitted'
        return self


    def build_matrix_for_prediction(self, X):

        nfolds = 5
        intermediate_answers = dict()
        for primary_clf in self.primary_clfs:
            if self.mode_ == 'BINARY':
                intermediate_answers[primary_clf['name']] = np.zeros((X.shape[0],))
            else:
                intermediate_answers[primary_clf['name']] = np.zeros((len(self.classes_), X.shape[0]))
            if self.mode_ == 'BINARY':
                intermediate_answers_skf = np.empty((nfolds, X.shape[0]))
            else:
                assert self.mode_ == 'MULTI'
                intermediate_answers_skf = [np.empty((nfolds, X.shape[0])) for class_idx in range(len(self.classes_))]
            for i in range(nfolds):
                if self.primary_is_proba:
                    if 'predict_proba' in dir(primary_clf['clfs'][i]):
                        probas = primary_clf['clfs'][i].predict_proba(X)
                        if self.mode_ == 'BINARY':
                            intermediate_answers_skf[i, :] = map(operator.itemgetter(1), probas)
                        else:
                            assert self.mode_ == 'MULTI'
                            for class_idx in range(len(self.classes_)):
                                intermediate_answers_skf[class_idx][i, :] = map(operator.itemgetter(class_idx), probas)
                    elif 'decision_function' in dir(primary_clf['clfs'][i]):
                        scores = primary_clf['clfs'][i].decision_function(X)
                        if self.mode_ == 'BINARY':
                            assert type(scores[0]) is np.float64
                            intermediate_answers_skf[i, :] = scores
                        else:
                            assert self.mode_ == 'MULTI'
                            assert type(scores[0]) is np.ndarray
                            for class_idx in range(len(self.classes_)):
                                intermediate_answers_skf[class_idx][i, :] = map(operator.itemgetter(class_idx), scores)
                    else:
                        print >> sys.stderr, 'Error in StackingClassifier: parameter "primary_is_proba" is True, however one of primary classifiers has neither "predict_proba" method nor "decision_function". This classifier is "{}"\n'.format(type(primary_clf['clfs'][i]))
                        exit(1)
                else:
                    intermediate_answers_skf[i, :] = primary_clf['clfs'][i].predict(X)

            if self.mode_ == 'BINARY':
                intermediate_answers[primary_clf['name']][:] = intermediate_answers_skf.mean(axis=0)
            else:
                assert self.mode_ == 'MULTI'
                if self.primary_answers_choosing_method == 'MEANS':
                    for class_idx in range(len(self.classes_)):
                        #print 'intermediate_answers_skf[class_idx].shape', intermediate_answers_skf[class_idx].shape
                        intermediate_answers[primary_clf['name']][class_idx][:] = intermediate_answers_skf[class_idx].mean(axis = 0)
                elif self.primary_answers_choosing_method == 'BEST_CLF':
                    for object_idx in range(X.shape[0]):
                        sorted_values = []
                        for class_idx in range(len(self.classes_)):
                            values = []
                            for fold_idx in range(nfolds):
                                values.append(intermediate_answers_skf[class_idx][fold_idx][object_idx])
                            sorted_values.append(sorted(values))
                        tuple_scores = []
                        max_score = None
                        max_score_tuple = None
                        for fold_idx in range(nfolds):
                            score = 0
                            for class_idx in range(len(self.classes_)):
                                curr_sorted_values = sorted_values[class_idx]
                                val = intermediate_answers_skf[class_idx][fold_idx][object_idx]
                                for i in range((nfolds + 1) / 2):
                                    if val == curr_sorted_values[i] or \
                                       val == curr_sorted_values[nfolds - 1 - i]:
                                        score += 10 ** i
                            tuple_scores.append(score)
                            if max_score is None or score > max_score:
                                max_score = score
                                max_score_tuple = [intermediate_answers_skf[class_idx][fold_idx][object_idx] for class_idx in range(len(self.classes_))]
                        for class_idx in range(len(self.classes_)):
                            intermediate_answers[primary_clf['name']][class_idx][object_idx] = max_score_tuple[class_idx]



                else:
                    assert self.primary_answers_choosing_method == 'RANDOM_CLF'
                    assert False
                    pass # ToDo: implement
                #print "intermediate_answers[primary_clf['name']].shape in build_matrix_for_prediction", intermediate_answers[primary_clf['name']].shape
        if self.mode_ == 'BINARY' or not self.primary_is_proba:
            features = [csr_matrix(intermediate_answers[primary_clf['name']].reshape(-1, 1)) for primary_clf in self.primary_clfs]
        else:
            assert self.mode_ == 'MULTI'
            features = [csr_matrix(intermediate_answers[primary_clf['name']][class_idx].reshape(-1, 1)) for primary_clf in self.primary_clfs for class_idx in range(len(self.classes_))]
        #print 'len(features) in build_matrix_for_prediction', len(features)
        #print 'features[0].shape in build_matrix_for_prediction', features[0].shape
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
                  'classes_': self.classes_,
                  'mode_': self.mode_,
                  'version': self.version,
                  'primary_answers_choosing_method': self.primary_answers_choosing_method}
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
            if param == 'mode_':
                if value is not None:
                    self.mode_ = value
            if param == 'version':
                if value is not None:
                    self.version = value
            if param == 'primary_answers_choosing_method':
                if value is not None:
                    self.primary_answers_choosing_method = value
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


