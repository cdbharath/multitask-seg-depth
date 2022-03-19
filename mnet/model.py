import torch
import torch.nn as nn
import math
from backbone import MultiFullyConnectedBackbone

class Encoder:
    def __init__(self):
        pass

class Decoder:
    def __init__(self):
        pass

class Fc:
    def __init__(self):
        pass

def net():
    encoder = Encoder()
    decoder = Decoder()
    fc = Fc()
    multinet = MultiFullyConnectedBackbone(encoder, decoder, tasks, fc)
    return multinet