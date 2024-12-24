# observation.py
import numpy as np

class DummyImage:
    def __init__(self):
        self.H = 256
        self.W = 256
    
    def image(self, cam_idx):
        x = np.arange(self.H)[:, None]
        y = np.arange(self.W)[None, :]
        checker = ((x + y) % 32 < 16).astype(np.uint8) * 255
        img = np.stack([checker] * 3, axis=-1)
        return img

class DummyObs:
    def __init__(self):
        self.state = np.zeros(7, dtype=np.float32)
        self._image = DummyImage()
        self.prev = None
        self._dict = {}  # Internal dictionary to support dict-like behavior
    
    def image(self, cam_idx):
        return self._image.image(cam_idx)
    
    # Add dictionary-like interface
    def keys(self):
        return self._dict.keys()
    
    def __getitem__(self, key):
        return self._dict[key]
    
    def __setitem__(self, key, value):
        self._dict[key] = value