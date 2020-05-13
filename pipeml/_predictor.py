# -*- coding: utf-8 -*-
"""
Created on Tue May 12 22:20:36 2020

@author: yoelr
"""


class Predictor:
    __slots__ = ('function', 'fit', 'metric')
    
    def __init__(self, metric, function, fit=None):
        self.metric = metric
        self.function = function
        self.fit = fit or function.fit
      
    def __call__(self, xs):
        return self.function(xs)
    
    def __repr__(self):
        return f"<{type(self).__metric__}: {self.metric}>"