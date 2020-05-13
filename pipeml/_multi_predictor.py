# -*- coding: utf-8 -*-
"""
Created on Tue May 12 21:52:28 2020

@author: yoelr
"""
import numpy as np
from ._predictor import Predictor

__all__ = ('MultiPredictor',)

class MultiPredictor:
    __slots__ = ('predictors',)
    
    def __init__(self, predictors):
        isa = isinstance
        for i in predictors:
            assert isa(i, Predictor), "all elemenents must be a Predictor object"
        self.predictors = predictors
        
    def fit(self, X, y):
        for predictor, yi in zip(self.predictors, y.transpose()): predictor.fit(X, yi)
        
    def __call__(self, xs):
        return np.array([i(xs) for i in self.predictors], float)
    
    @property
    def metrics(self):
        return [i.metric for i in self.predictors]
    
    def __repr__(self):
        names = ', '.join(self.metrics)
        return f"<{type(self).__name__}: {names}>"
        