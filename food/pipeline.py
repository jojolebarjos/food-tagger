# -*- coding: utf-8 -*-


from .layer import CompoundLayer
from .entity import EntityLayer
from .ingredient import IngredientLayer
from .pos import PosLayer
from .token import TokenLayer


# Default pipeline
def get_default_layer():
    return CompoundLayer(
        TokenLayer(),
        PosLayer(),
        EntityLayer(),
        IngredientLayer()
    )
