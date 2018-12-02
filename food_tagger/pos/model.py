# -*- coding: utf-8 -*-


import os
import pycrfsuite

from .dataset import get_samples
from ..layer import Layer


# Model file
HERE = os.path.dirname(os.path.realpath(__file__))
MODEL_CRFSUITE = os.path.join(HERE, 'model.crfsuite')


# Generate features for Part-of-Speech tagger
def extract_features(tokens):
    # TODO use lemma at some point, maybe in a dedicated layer?
    
    # Prepare features
    # TODO improve features
    words = [token.lower() for token in tokens]

    # Collect features
    result = []
    for i in range(len(tokens)):
        features = {}
        features['b'] = 1.0
        
        # Previous token
        if i > 0:
            features['-w' + words[i - 1]] = 1.0
        else:
            features['s'] = 1.0
        
        # Current token
        features['w' + words[i]] = 1.0
        
        # Next token
        if i < len(tokens) - 1:
            features['+w' + words[i + 1]] = 1.0
        else:
            features['e'] = 1.0
        
        # Ready
        result.append(features)
    return result


# Retrain PoS model
def train(max_iterations=100):
    
    # Create trainer
    trainer = pycrfsuite.Trainer(verbose=True)
    trainer.set_params({
        'c1': 1.0,
        'c2': 1e-3,
        'max_iterations': max_iterations,
        'feature.possible_transitions': True
    })
    
    # Generate training samples
    for tokens, tags in get_samples(test=False):
        features = extract_features(tokens)
        trainer.append(features, tags)

    # Train
    trainer.train(MODEL_CRFSUITE)
    
    # TODO evaluate


# Part-of-Speech layer
class PosLayer(Layer):
    def __init__(self):
        tagger = pycrfsuite.Tagger()
        tagger.open(MODEL_CRFSUITE)
        self.tagger = tagger
    
    def apply(self, sample):
        sample = dict(sample)
        tokens = sample['tokens']
        features = extract_features(tokens)
        pos_tags = self.tagger.tag(features)
        sample['pos_tags'] = pos_tags
        return sample
