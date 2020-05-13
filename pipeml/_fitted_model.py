# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:37:59 2020

@author: yoelr
"""
import numpy as np
from ._rescaler import Rescaler
from ._denormalizer import Denormalizer
from ._predictor import Predictor
from ._multi_predictor import MultiPredictor


__all__ = ('FittedModel',)

def assert_within_bounds(name, lb, xi, ub):
    assert (lb < xi < ub).all(), f"{name} ({xi:.5g}) is not within bounds {(lb, ub)}"

class FittedModel:
    __slots__ = ('_predictor', 'preprocessor', 'postprocessor', 'bounds', 'features')
    
    def __init__(self, predictor, preprocessor, postprocessor, bounds=None, features=None):
        self.predictor = predictor
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.bounds = bounds
        self.features = features
        
    def __call__(self, xs, check_bounds=True):
        xs = np.asarray(xs, float)
        if check_bounds: self.check_bounds(xs)
        xs = self.preprocessor(xs)
        ys = self.predictor.predict(xs)
        return self.postprocessor(ys)
    
    @property
    def predictor(self):
        return self._predictor
    @predictor.setter
    def predictor(self, predictor):
        assert isinstance(predictor, (Predictor, MultiPredictor))
        self._predictor = predictor
    
    @property
    def metrics(self):
        predictor = self.predictor
        if isinstance(predictor, Predictor):
            return [predictor.metric]
        elif isinstance(predictor, MultiPredictor):
            return predictor.metrics
        else:
            raise AttributeError('could not find metric names')
    
    @classmethod
    def from_dfs(cls, Xdf, ydf, ML=None, bounds=None, features=None):
        X = Xdf.values
        y = ydf.values
        preprocessor = Rescaler.from_data(X)
        X = preprocessor(X)
        postprocessor = Denormalizer.from_data(y)
        y = postprocessor.scale(y)
        if not ML: from sklearn.svm import SVR as ML
        if ydf.ndim == 1:
            predictor = Predictor(ydf.name, ML())
        else:
            def create_predictor(metric):
                ml = ML()
                ml.metric = metric
                return ml
            predictors = [create_predictor(col) for col in ydf]
            predictor = MultiPredictor(predictors)
        predictor.fit(X, y)
        if not features: features = tuple(Xdf)
        self = cls(predictor, preprocessor, postprocessor, bounds, features)
        return self
    
    def check_bounds(self, xs):
        bounds = self.bounds
        if bounds is None: return
        names = self.features
        if names:
            for name, xi, (lb, ub) in zip(names, xs.transpose(), bounds):
                assert_within_bounds(name, lb, xi, ub)
        else:
            for i, (xi, (lb, ub)) in enumerate(zip(xs.transpose(), bounds)):
                name = f"Feature #{i}"
                assert_within_bounds(name, lb, xi, ub)
            
    def __repr__(self):
        newline = "\n" + " " * len(" metrics: ")
        metric_info = newline.join([str(i) for i in self.metrics])
        features = self.features
        if features:
            newline = "\n" + " " * len(" features: ")
            feature_info = newline.join([str(i) for i in features])
        else:
            feature_info = "?"
        return (f"{type(self).__name__}:\n"
                f" metrics: {metric_info}\n"
                f" features: {feature_info}")