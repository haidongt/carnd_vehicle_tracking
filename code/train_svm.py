import json
import pickle
import time

import numpy as np
import cv2

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.model_selection import train_test_split

from skimage.feature import hog

N_JOBS = 1
SAMPLE_SIZE = 10


class TrainPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, cs_bins=None, cs_cspace=None, chist_cspace=None, chist_bins=None,
        hog_cspace=None, pix_per_cell=None, cells_per_block=None, orient=None):
        self.cs_bins = cs_bins
        self.cs_cspace = cs_cspace
        self.chist_cspace = chist_cspace
        self.chist_bins = chist_bins
        self.hog_cspace = hog_cspace
        self.pix_per_cell = pix_per_cell
        self.cells_per_block = cells_per_block
        self.orient = orient
        self.clf = None
        self.standard_scaler = None

    def fit(self, data, y=None):
        return self

    def score(self, images, y):
        X = self.get_features(images)
        self.standard_scaler = StandardScaler()
        X = self.standard_scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=7)

        self.clf = LinearSVC()

        self.clf.fit(X_train, y_train)

        score = self.clf.score(X_test, y_test)
        print('score:', score)
        return score

    def get_features(self, images):
        cs_features = self._spacial_binning(
            self._transform_color(images, self.cs_cspace),
            self.cs_bins)

        chist_features = self.color_histogram(self._transform_color(images, self.chist_cspace))

        hog_features = self._hog(self._transform_color(images, self.hog_cspace))

        X = np.concatenate([cs_features, chist_features, hog_features], axis=1)
        return X

    def _spacial_binning(self, images, bins):
        spatial = np.zeros((len(images), bins * bins * images.shape[-1]))

        for i, img in enumerate(images):
            spatial[i] = cv2.resize(img, (bins, bins)).ravel()

        return spatial

    def _transform_color(self, images, cspace):
        if len(images[0]) == 0:
            return images

        if cspace == 'GRAY':
            result = np.zeros((*images.shape[:-1], 1))
        else:
            result = np.zeros(images.shape)

        for i, img in enumerate(images):
            if cspace == 'RGB':
                result[i] = (np.copy(img))
            elif cspace == 'HSV':
                result[i] = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                result[i] = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                result[i] = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                result[i] = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                result[i] = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            elif cspace == 'LAB':
                result[i] = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            elif cspace == 'GRAY':
                result[i] = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), axis=-1)
        return result

    def color_histogram(self, images):
        if len(images[0]) == 0:
            return images

        result = np.zeros((len(images), self.chist_bins * images.shape[-1]))

        for i, img in enumerate(images):
            hists = np.zeros((img.shape[-1], self.chist_bins))
            for ch in range(img.shape[-1]):
                hists[ch] = np.histogram(img[:, :, ch], bins=self.chist_bins, range=(0, 256))[0]

            result[i] = hists.ravel()

        return result

    def _hog(self, images):
        if len(images[0]) == 0:
            return images

        hogs = np.zeros(
            (len(images), 3 * self._hog_feature_size(images[0], self.pix_per_cell, self.cells_per_block, self.orient)))

        for i, img in enumerate(images):
            hogs[i] = self._hog_features(img, self.orient, self.pix_per_cell, self.cells_per_block, vis=False)

        return hogs

    def _hog_feature_size(self, img, pix_per_cell, cells_per_block, orient):

        if type(img) is np.ndarray and 1 < len(img.shape) < 4:
            span_y, span_x = img.shape[0], img.shape[1]
        elif type(img) in (tuple, list) and len(img) == 2:
            span_y, span_x = img
        else:
            span_y, span_x = img, img

        n_cells_x = int(np.floor(span_x // pix_per_cell))
        n_cells_y = int(np.floor(span_y // pix_per_cell))
        n_blocks_x = (n_cells_x - cells_per_block) + 1
        n_blocks_y = (n_cells_y - cells_per_block) + 1

        return n_blocks_y * n_blocks_x * cells_per_block * cells_per_block * orient

    def _hog_features(self, img, orient=9, pix_per_cell=8, cells_per_block=2, vis=False):

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        features = np.zeros((img.shape[2], self._hog_feature_size(img, pix_per_cell, cells_per_block, orient)))

        if vis:
            hog_image = np.zeros(img.shape, dtype=np.float32)

        for ch in range(img.shape[2]):
            hog_result = hog(img[:, :, ch], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                             cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=True,
                             visualise=vis, feature_vector=True)

            if vis:
                features[ch] = hog_result[0]
                hog_image[:, :, ch] = hog_result[1]
            else:
                features[ch] = hog_result

        features = features.ravel()

        if vis:
            return features, hog_image
        else:
            return features


if __name__ == '__main__':
    np.random.seed(5)

    with open('../data.p', 'rb') as f:
        data = pickle.load(f)

    X_train, y_train = data['train']
    X_val, y_val = data['val']
    X_test, y_test = data['test']

    X = np.concatenate([X_train, X_val, X_test])
    y = np.concatenate([y_train, y_val, y_test])

    X, y = shuffle(X, y, random_state=7)


    params = {
    'cs_bins': [32],
    'cs_cspace': ['LAB'],
    'chist_cspace': ['HLS'],
    'chist_bins': [32],
    'hog_cspace': ['LAB'],
    'pix_per_cell': [8],
    'cells_per_block': [2],
    'orient': [18]
    }

    pipeline = TrainPipeline(*params)


    cls = GridSearchCV(pipeline, params, cv=2, n_jobs=N_JOBS, verbose=3)
    print('Begin training')
    t = time.time()
    cls.fit(X, y)
    t2 = time.time()
    print('Finished training after ', t2 - t, ' seconds')


    print('Best params: ', cls.best_params_)

    with open('../svm_final.p', 'wb') as f:
        pickle.dump(cls.best_estimator_, f)


    best_pipeline = TrainPipeline(**cls.best_params_)
    best_pipeline.score(X, y)
    clf = best_pipeline.clf
    scaler = best_pipeline.standard_scaler

    features = best_pipeline.get_features(X)
    features = scaler.transform(features)
    print(clf.score(features, y))
    print(clf.predict(features))
    print(clf.decision_function(features))


    with open('../svm_model.p', 'wb') as f:
        pickle.dump({
            'pipeline': best_pipeline,
            'scaler': scaler,
            'clf': clf
            }, f)

    '''
    print('testing saved model ')
    with open('../svm_model.p', 'rb') as f:
        model = pickle.load(f)

    features = model['pipeline'].get_features(X)
    features = model['scaler'].transform(features)
    print(model['clf'].score(features, y))
    print(model['clf'].predict(features))
    '''