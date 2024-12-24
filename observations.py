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
    
    def image(self, cam_idx):
        return self._image.image(cam_idx)