# --- context manager for replacing MLP sublayers with transcoders ---
import torch

class TranscoderWrapper(torch.nn.Module):
    def __init__(self, transcoder):
        super().__init__()
        self.transcoder = transcoder
    def forward(self, x):
        return self.transcoder(x)[0]

class TranscoderReplacementContext:
    def __init__(self, model, transcoders):
        self.layers = [t.cfg.hook_point_layer for t in transcoders]
        self.original_mlps = [ model.blocks[i].mlp for i in self.layers ]
        
        self.transcoders = transcoders
        #self.layers = layers
        self.model = model
    
    def __enter__(self):
        for transcoder in self.transcoders:
           self.model.blocks[transcoder.cfg.hook_point_layer].mlp = TranscoderWrapper(transcoder)

    def __exit__(self, exc_type, exc_value, exc_tb):
        for layer, mlp in zip(self.layers, self.original_mlps):
            self.model.blocks[layer].mlp = mlp

class ZeroAblationWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x*0.0

class ZeroAblationContext:
    def __init__(self, model, layers):
        self.original_mlps = [ model.blocks[i].mlp for i in layers ]
        
        self.layers = layers
        self.model = model
    
    def __enter__(self):
        for layer in self.layers:
           self.model.blocks[layer].mlp = ZeroAblationWrapper()

    def __exit__(self, exc_type, exc_value, exc_tb):
        for layer, mlp in zip(self.layers, self.original_mlps):
            self.model.blocks[layer].mlp = mlp