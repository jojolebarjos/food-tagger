# -*- coding: utf-8 -*-


# Simple abstraction over modules
class Layer:
    def apply(self, sample):
        raise NotImplementedError()


# Compound layer
class CompoundLayer(Layer):
    def __init__(self, *layers):
        self.layers = layers
    
    def apply(self, sample):
        for layer in self.layers:
            sample = layer.apply(sample)
        return sample
