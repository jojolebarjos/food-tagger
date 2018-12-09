# -*- coding: utf-8 -*-


import regex as re

from .layer import Layer


TOKEN_REGEX = re.compile(r'\s*(\p{L}+|\d+|\S)', re.UNICODE)


# Use simple rules to tokenize text
def iter_token(text):
    index = 0
    while True:
        match = TOKEN_REGEX.match(text, index)
        if not match:
            break
        token = match.group(1)
        start = match.start(1)
        end = match.end(1)
        yield token, start, end
        index = end


# List-based tokenization
def tokenize(text):
    return [token for token, _, _ in iter_token(text)]


# Tokenization layer
class TokenLayer(Layer):
    def apply(self, sample):
        if type(sample) is str:
            text = sample
            sample = {'text': text}
        else:
            sample = dict(sample)
            text = sample['text']
        tokens = []
        spans = []
        for token, start, end in iter_token(text):
            tokens.append(token)
            spans.append((start, end))
        sample['tokens'] = tokens
        sample['spans'] = spans
        return sample
        
