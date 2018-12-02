# -*- coding: utf-8 -*-


import io
import os

from ..token import iter_token


# Resources
HERE = os.path.dirname(os.path.realpath(__file__))
TRAIN_CONLLU = os.path.join(HERE, 'UD_English', 'en-ud-train.conllu')
TEST_CONLLU = os.path.join(HERE, 'UD_English', 'en-ud-test.conllu')


# Import CoNLL-U Part-of-Speech dataset, with retokenization
def from_conllu(file):
    samples = []
    text = ''
    text_tags = []
    
    # For each line
    for line in file:
        line = line.rstrip()
        
        # Empty line marks end-of-sample
        if len(line) == 0:
            tokens = []
            tags = []
            for token, start, end in iter_token(text):
                tag = text_tags[start]
                tokens.append(token)
                tags.append(tag)
            if len(tokens) > 0:
                sample = (tokens, tags)
                samples.append(sample)
            text = ''
            text_tags = []
            continue
        
        # Ignore comments
        if line[0] == '#':
            continue
        
        # Acquire token info
        parts = line.split('\t')
        token = parts[1]
        tag = parts[3]
        misc = parts[9].split('|')
        space_after = 'SpaceAfter=No' not in misc
        
        # Accumulate text
        if space_after:
            token += ' '
        text += token
        text_tags.extend([tag] * len(token))
    
    # Ready
    return samples


# Get complete Part-of-Speech dataset
def get_samples(test=False):
    path = TEST_CONLLU if test else TRAIN_CONLLU
    with io.open(path, 'r', newline='\n', encoding='utf-8') as file:
        samples = from_conllu(file)
    # TODO also use task-specific samples
    return samples
