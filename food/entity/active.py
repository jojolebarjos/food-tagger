# -*- coding: utf-8 -*-


import io
import os
import random

from .dataset import get_samples, from_excel, to_excel, to_tab, TRAIN_TXT, TEST_TXT
from .model import EntityLayer, train
from ..layer import CompoundLayer
from ..pos import PosLayer
from ..token import TokenLayer


# Resources
HERE = os.path.dirname(os.path.realpath(__file__))
SAMPLES_TXT = os.path.join(HERE, 'samples.txt')
SAMPLES_XLSX = os.path.join(HERE, 'samples.xlsx')


# Select next samples to annotate
def generate_samples(count=5, oversampling=10, path=SAMPLES_XLSX):
    
    # Load unannotated samples
    with io.open(SAMPLES_TXT, 'r', encoding='utf-8', newline='\n') as file:
        unannotated_samples = {line.strip() for line in file}
    
    # Load annotated samples
    annotated_samples = {text for split in [False, True] for text, _, _, _ in get_samples(split)}
    
    # Keep random subset
    samples = list(unannotated_samples.difference(annotated_samples))
    random.shuffle(samples)
    samples = samples[:count * oversampling]
    
    # Load model
    layer = CompoundLayer(
        TokenLayer(),
        PosLayer(),
        EntityLayer()
    )
    
    # Compute prediction and confidence
    samples = [layer.apply(sample) for sample in samples]
    
    # Keep less confident samples
    # TODO is this the correct metric? should maybe use min(sample['entity_tags_probabilities'])
    samples.sort(key=lambda sample: sample['entity_tags_probability'])
    samples = samples[:count]
    
    # Convert to expected format
    samples = [(sample['text'], sample['tokens'], sample['pos_tags'], sample['entity_tags']) for sample in samples]
    
    # Export samples to Excel file, for easy annotation
    with io.open(path, 'wb') as file:
        to_excel(file, samples)


# Add samples to training set
def import_samples(path=SAMPLES_XLSX):
    
    # Import new samples
    with io.open(path, 'rb') as file:
        samples = from_excel(file)
    
    # Split by half, for training and testing sets
    train_samples = []
    test_samples = []
    for sample in samples:
        if random.random() >= 0.5:
            train_samples.append(sample)
        else:
            test_samples.append(sample)
    
    # Append to dataset
    for path, samples in [(TRAIN_TXT, train_samples), (TEST_TXT, test_samples)]:
        with io.open(path, 'a', encoding='utf-8', newline='\n') as file:
            to_tab(file, samples)
