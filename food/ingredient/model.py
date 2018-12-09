# -*- coding: utf-8 -*-


import os

from .dataset import get_samples
from ..layer import Layer


# Train model
def train(max_iterations=50):
    pass


# Ingredient classification layer
class IngredientLayer(Layer):
    def __init__(self):
        pass
    
    def apply(self, sample):
        # TODO ingredient classification
        return sample
