# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:37:59 2020

@author: yoelr
"""
import numpy as np

class FittedModel:
    __slots__ = ('predictor', 'preprocessor', 'bounds', 'names')
    
    def __init__(self, predictor, preprocessor, bounds, names):
        self.predictor = predictor
        self.preprocessor = preprocessor
        self.bounds = bounds
        self.names = names
        
    def __call__(self, xs, check_bounds=True):
        xs = np.asarray(xs, float)
        if check_bounds: self.check_bounds(xs)
        xs = self.preprocessor(xs)
        return self.predictor(xs)
    
    def check_bounds(self, xs):
        for name, xi, (lb, ub) in zip(self.names, xs.transpose(), self.bounds):
            assert (lb < xi < ub).all(), f"{name} ({xi:.5g} is not within bounds {(lb, ub)}"
            
    def __repr__(self):
        return f"<{type(self).__name__}: {', '.join(self.names)}>"