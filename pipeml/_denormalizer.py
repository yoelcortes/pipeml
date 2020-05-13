# -*- coding: utf-8 -*-
"""
Created on Tue May 12 23:02:00 2020

@author: yoelr
"""
__all__ = ('Denormalizer',)

class Denormalizer:
    __slots__ = ('width', 'minimum')
    
    def __init__(self, width, minimum):
        self.width = width
        self.minimum = minimum

    @classmethod
    def from_data(cls, X, axis=None):
        if axis is None:
            axis = list(range(X.ndim))
            axis.remove(axis[-1])
            axis = tuple(axis)
        maximum = X.max(axis=axis, keepdims=True)
        minimum = X.min(axis=axis, keepdims=True)
        width = maximum - minimum
        return cls(width, minimum)
    
    def scale(self, X):
        return (X - self.minimum) / self.width
    
    def unscale(self, X):
        return (X * self.width) + self.minimum
    
    __call__ = unscale
    
    def __repr__(self):
        return f"<{type(self).__name__}>"
    