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
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

from skimage.feature import hog

N_JOBS = 1
SAMPLE_SIZE = 10

class ColorSpaceConverter(BaseEstimator, TransformerMixin):
    def __init__(self, cspace='RGB', single_channel=None):
        self.single_channel = single_channel
        self.cspace = cspace

    def fit(self, data, y=None):
        return self

    def transform(self, images):
        if len(images[0]) == 0:
            return images

        if self.single_channel is not None or self.cspace == 'GRAY':
            result = np.zeros((*images.shape[:-1], 1))
        else:
            result = np.zeros(images.shape)

        for i, img in enumerate(images):
            if self.cspace == 'RGB':
                result[i] = (np.copy(img))
            elif self.cspace == 'HSV':
                result[i] = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif self.cspace == 'LUV':
                result[i] = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif self.cspace == 'HLS':
                result[i] = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif self.cspace == 'YUV':
                result[i] = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif self.cspace == 'YCrCb':
                result[i] = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            elif self.cspace == 'LAB':
                result[i] = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            elif self.cspace == 'GRAY':
                result[i] = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), axis=-1)
        return result


class SpatialBining(BaseEstimator, TransformerMixin):
    def __init__(self, bins=32):
        self.bins = bins

    def fit(self, data, y=None):
        return self

    def transform(self, images):
        spatial = np.zeros((len(images), self.bins * self.bins * images.shape[-1]))

        for i, img in enumerate(images):
            spatial[i] = cv2.resize(img, (self.bins, self.bins)).ravel()

        return spatial


class ColorHistogram(BaseEstimator, TransformerMixin):
    def __init__(self, bins=32, bins_range=(0, 256)):

        self.bins_range = bins_range
        self.bins = bins

    def fit(self, data, y=None):
        return self

    def transform(self, images):
        if len(images[0]) == 0:
            return images

        result = np.zeros((len(images), self.bins * images.shape[-1]))

        for i, img in enumerate(images):
            hists = np.zeros((img.shape[-1], self.bins))
            for ch in range(img.shape[-1]):
                hists[ch] = np.histogram(img[:, :, ch], bins=self.bins, range=self.bins_range)[0]

            result[i] = hists.ravel()

        return result


class HogExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, orient=9, pix_per_cell=8, cells_per_block=2):
        self.pix_per_cell = pix_per_cell
        self.cells_per_block = cells_per_block
        self.orient = orient

    def fit(self, data, y=None):
        return self

    def transform(self, images):
        if len(images[0]) == 0:
            return images

        hogs = np.zeros(
            (len(images), 3 * self.hog_feature_size(images[0], self.pix_per_cell, self.cells_per_block, self.orient)))

        for i, img in enumerate(images):
            hogs[i] = self.hog_features(img, self.orient, self.pix_per_cell, self.cells_per_block, vis=False)

        return hogs

    def hog_feature_size(self, img, pix_per_cell, cells_per_block, orient):

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

    def hog_features(self, img, orient=9, pix_per_cell=8, cells_per_block=2, vis=False):

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        features = np.zeros((img.shape[2], self.hog_feature_size(img, pix_per_cell, cells_per_block, orient)))

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

    X_train = np.concatenate([X_train, X_val])
    y_train = np.concatenate([y_train, y_val])

    X_train, y_train = shuffle(X_train, y_train, random_state=7)

    print('Train size:', len(y_train))


    sb_csc = ColorSpaceConverter()
    spatial_bining = SpatialBining()
    sb_pipeline = Pipeline([("sb_csc", sb_csc),
                            ("spatial_bining", spatial_bining),
                            ("sb_minmax", MinMaxScaler(feature_range=(0, 1)))])

    chist_csc = ColorSpaceConverter()
    color_histogram = ColorHistogram()
    chist_pipeline = Pipeline([("chist_csc", chist_csc),
                               ("color_histogram", color_histogram),
                               ("chist_minmax", MinMaxScaler(feature_range=(0, 1)))])

    hoh_csc = ColorSpaceConverter()
    hog_extractor = HogExtractor()
    hog_pipeline = Pipeline([("hog_csc", hoh_csc),
                             ("hog_extractor", hog_extractor),
                             ("hog_minmax", MinMaxScaler(feature_range=(0, 1)))])

    features = FeatureUnion([("hog", hog_pipeline), ('chist', chist_pipeline), ('sb', sb_pipeline)], n_jobs=1)

    clf = LinearSVC()
    pipeline = Pipeline([('features', features),
                         ('clf', clf)])

    with open('grid_config.json') as data_file:
        params = json.load(data_file)

    cls = GridSearchCV(pipeline, params, cv=2, n_jobs=N_JOBS, verbose=3, scoring='roc_auc', )

    print('Begin training')
    t = time.time()
    cls.fit(X_train, y_train)
    t2 = time.time()
    print('Finished training after ', t2 - t, ' seconds')

    with open('../models/svm_final.p', 'wb') as f:
        pickle.dump(cls.best_estimator_, f)

    with open('../models/gridsearch_final.p', 'wb') as f:
        pickle.dump(cls, f)

    print('Best params: ', cls.best_params_)
    print('Best auc roc score: ', cls.best_score_)
    print('Train Accuracy of SVC = ', cls.best_estimator_.score(X_train, y_train))
    print('Test Accuracy of SVC = ', cls.best_estimator_.score(X_test, y_test))

    predictions = cls.predict(X_test)
    print(classification_report(y_test, predictions, target_names=['no car', 'car']))
