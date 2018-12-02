# -*- coding: utf-8 -*-


import os
import pycrfsuite

from .dataset import get_samples
from ..layer import Layer
from ..pos.model import extract_features as pos_extract_features


# Model file
HERE = os.path.dirname(os.path.realpath(__file__))
MODEL_CRFSUITE = os.path.join(HERE, 'model.crfsuite')


# Generate features for entity tagger
def extract_features(tokens, pos_tags):
    result = pos_extract_features(tokens)
    for i in range(len(tokens)):
        features = result[i]
        if i > 0:
            features['-p' + pos_tags[i - 1]] = 1.0
        features['p' + pos_tags[i]] = 1.0
        if i < len(tokens) - 1:
            features['+p' + pos_tags[i + 1]] = 1.0
    return result


# Train model
def train(max_iterations=50):
    
    # Create trainer
    trainer = pycrfsuite.Trainer(verbose=True)
    trainer.set_params({
        'c1': 1.0,
        'c2': 1e-3,
        'max_iterations': max_iterations,
        'feature.possible_transitions': True
    })
    
    # Generate training samples
    for _, tokens, pos_tags, entity_tags in get_samples(test=False):
        features = extract_features(tokens, pos_tags)
        trainer.append(features, entity_tags)

    # Train
    trainer.train(MODEL_CRFSUITE)
    
    # TODO evaluate


# Ingredient entity layer
class EntityLayer(Layer):
    def __init__(self):
        tagger = pycrfsuite.Tagger()
        tagger.open(MODEL_CRFSUITE)
        self.tagger = tagger
    
    def apply(self, sample):
        sample = dict(sample)
        tokens = sample['tokens']
        pos_tags = sample['pos_tags']
        features = extract_features(tokens, pos_tags)
        entity_tags = self.tagger.tag(features)
        entity_tags_probabilities = [self.tagger.marginal(entity_tags[i], i) for i in range(len(tokens))]
        entity_tags_probability = self.tagger.probability(entity_tags)
        sample['entity_tags'] = entity_tags
        sample['entity_tags_probabilities'] = entity_tags_probabilities
        sample['entity_tags_probability'] = entity_tags_probability
        return sample
