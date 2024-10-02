# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .deco import build as build_deco

def build_model(args):
    return build_deco(args)
