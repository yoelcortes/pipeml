# -*- coding: utf-8 -*-
"""
Created on Tue May 12 22:20:36 2020

@author: yoelr
"""

__all__ = ('Predictor',)

class Predictor:
    __slots__ = ('predict', 'fit', 'metric')
    
    def __init__(self, metric, predict, fit):
        self.metric = metric
        self.predict = predict
        self.fit = fit
    
    def __repr__(self):
        return f"<{type(self).__metric__}: {self.metric}>"