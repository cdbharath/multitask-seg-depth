import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleTaskBackbone(nn.Module):
    '''
    Single Encoder - Single Decoder Architecture
    '''
    def __init__(self, encoder, decoder, tasks):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tasks = tasks

    def forward(self, x):
        return self.decoder(self.encoder(x))

class MultiDecoderBackbone(SingleTaskBackbone):
    '''
    Single Encoder - Multi Decoder Architecture
    '''
    def __init__(self, encoder, decoder, tasks):
        super().__init__(encoder, decoder, tasks)
    
    def forward(self, x):
        tasks = []
        for decoder in self.decoder:
            tasks.append(decoder(self.encoder(x)))
        
        return tasks

class MultiFullyConnectedBackbone(SingleTaskBackbone):
    '''
    Single Encoder - Single Decoder - Multi Fully Connected Network Architecture
    '''
    def __init__(self, encoder, decoder, tasks, fc):
        super().__init__(encoder, decoder, tasks)
        self.fc = fc
    
    def forward(self, x):
        tasks = []
        for fc in self.fc:
            tasks.append(fc(self.decoder(self.encoder(x))))
        
        return tasks
