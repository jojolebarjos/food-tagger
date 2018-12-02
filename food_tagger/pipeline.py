# -*- coding: utf-8 -*-


from .layer import CompoundLayer
from .pos import PosLayer
from .token import TokenLayer


def get_default_layer():
    return CompoundLayer(
        TokenLayer(),
        PosLayer()
    )
