# -*- coding: utf-8 -*-


import io
import os


# Local resources
HERE = os.path.dirname(os.path.realpath(__file__))
TRAIN_TXT = os.path.join(HERE, 'train.txt')
TEST_TXT = os.path.join(HERE, 'test.txt')


# Import tab separated dataset
def from_tab(file):
    samples = []
    text = None
    tokens = []
    pos_tags = []
    entity_tags = []
    
    # For each line
    for line in file:
        line = line.rstrip()
        
        # Empty line marks end-of-sample
        if len(line) == 0:
            if len(tokens) > 0:
                sample = (text, tokens, pos_tags, entity_tags)
                samples.append(sample)
            text = None
            tokens = []
            pos_tags = []
            entity_tags = []
            continue
        
        # Keep comments as text reference
        if line[0] == '#':
            if line.startswith('# text = '):
                text = line[9:]
            continue
        
        # Accumulate tokens
        _, token, pos_tag, entity_tag = line.split('\t')
        tokens.append(token)
        pos_tags.append(pos_tag)
        entity_tags.append(entity_tag)
    
    # Ready
    return samples


# Export tab separated dataset
def to_tab(file, samples):
    for text, tokens, pos_tags, entity_tags in samples:
        if text is not None:
            file.write(f'# text = {text}')
        for index, (token, pos_tag, entity_tag) in enumerate(zip(tokens, pos_tags, entity_tags)):
            file.write(f'{index + 1}\t{token}\t{pos_tag}\t{entity_tag}\n')
        file.write('\n')
    file.flush()


# Get entity set
def get_samples(test=False):
    path = TEST_TXT if test else TRAIN_TXT
    with io.open(path, 'r', encoding='utf-8', newline='\n') as file:
        return from_tab(file)
