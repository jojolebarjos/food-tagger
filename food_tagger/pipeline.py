# -*- coding: utf-8 -*-


from .layer import CompoundLayer
from .pos import PosLayer
from .token import TokenLayer
from .entity import EntityLayer


# Default pipeline
def get_default_layer():
    return CompoundLayer(
        TokenLayer(),
        PosLayer(),
        EntityLayer()
    )
