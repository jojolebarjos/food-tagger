# -*- coding: utf-8 -*-


import os
import pycrfsuite

from .dataset import get_samples
from ..layer import Layer
from ..pos.model import extract_features as pos_extract_features, MODEL_CRFSUITE as POS_MODEL_CRFSUITE


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
    
    # Load taggers
    pos_tagger = pycrfsuite.Tagger()
    pos_tagger.open(POS_MODEL_CRFSUITE)
    entity_tagger = pycrfsuite.Tagger()
    entity_tagger.open(MODEL_CRFSUITE)
    
    # Apply to datasets
    for name, split in [('Train', False), ('Test', True)]:
        pos_truth = []
        pos_prediction = []
        entity_truth = []
        entity_prediction = []
        entity_cascaded_prediction = []
        for _, tokens, pos_tags, entity_tags in get_samples(test=split):
            
            # Predict Part-of-Speech
            pos_features = pos_extract_features(tokens)
            predicted_pos_tags = pos_tagger.tag(pos_features)
            pos_truth.extend(pos_tags)
            pos_prediction.extend(predicted_pos_tags)
            
            # Predict entity tags, using true PoS tags
            entity_features = extract_features(tokens, pos_tags)
            predicted_entity_tags = entity_tagger.tag(entity_features)
            entity_truth.extend(entity_tags)
            entity_prediction.extend(predicted_entity_tags)
            
            # Predict entity tags, using predicted 
            entity_features = extract_features(tokens, predicted_pos_tags)
            predicted_entity_tags = entity_tagger.tag(entity_features)
            entity_cascaded_prediction.extend(predicted_entity_tags)
        
        # Compute accuracies
        # TODO precision, recall, F-1
        def accuracy(truth, prediction):
            if len(truth) == 0:
                return 0.0
            return 100.0 * sum(a == b for a, b in zip(truth, prediction)) / len(truth)
        pos_accuracy = accuracy(pos_truth, pos_prediction)
        entity_accuracy = accuracy(entity_truth, entity_prediction)
        entity_cascaded_accuracy = accuracy(entity_truth, entity_cascaded_prediction)
        print(f'{name} accuracies:')
        print(f'  PoS:        {pos_accuracy: 3.2f}')
        print(f'  Entity:     {entity_accuracy: 3.2f}')
        print(f'  PoS+Entity: {entity_cascaded_accuracy: 3.2f}')


# Isolate entities
def bio_to_spans(tags):
    spans = []
    labels = []
    index = 0
    while index < len(tags):
        if '-' in tags[index]:
            label = tags[index][2:]
            start = index
            index += 1
            while index < len(tags) and tags[index] == 'I-' + label:
                index += 1
            span = (start, index)
            spans.append(span)
            labels.append(label)
        else:
            index += 1
    return spans, labels


# Ingredient entity layer
class EntityLayer(Layer):
    def __init__(self):
        tagger = pycrfsuite.Tagger()
        tagger.open(MODEL_CRFSUITE)
        self.tagger = tagger
    
    # Apply model
    def apply(self, sample):
        tokens = sample['tokens']
        pos_tags = sample['pos_tags']
        features = extract_features(tokens, pos_tags)
        entity_tags = self.tagger.tag(features)
        entity_tags_probabilities = [self.tagger.marginal(entity_tags[i], i) for i in range(len(tokens))]
        entity_tags_probability = self.tagger.probability(entity_tags)
        entity_spans, entity_labels = bio_to_spans(entity_tags)
        
        # Pack
        sample = {
            **sample,
            'entity_tags': entity_tags,
            'entity_tags_probabilities': entity_tags_probabilities,
            'entity_tags_probability': entity_tags_probability,
            'entity_spans': entity_spans,
            'entity_labels': entity_labels
        }
        return sample
