import numpy as np
import copy

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
        # Create internal dictionary storage
        self._obs = {
            'state': np.zeros(7, dtype=np.float32),
            'enc_cam_0': self._create_encoded_image()
        }
        self.prev = None
    
    def _create_encoded_image(self):
        # Create a checkerboard pattern
        x = np.arange(256)[:, None]
        y = np.arange(256)[None, :]
        checker = ((x + y) % 32 < 16).astype(np.uint8) * 255
        img = np.stack([checker] * 3, axis=-1)
        return img.tobytes()  # Convert to bytes to simulate encoded image
    
    @property
    def state(self):
        return self._obs['state']
    
    def image(self, cam_idx):
        # Return RGB image of shape (H, W, 3)
        x = np.arange(256)[:, None]
        y = np.arange(256)[None, :]
        checker = ((x + y) % 32 < 16).astype(np.uint8) * 255
        return np.stack([checker] * 3, axis=-1)
    
    def keys(self):
        return self._obs.keys()
    
    def __getitem__(self, key):
        return self._obs[key]
    
    def to_dict(self):
        return copy.deepcopy(self._obs)
    
    @classmethod
    def from_dict(cls, obs_dict):
        obs = cls()
        obs._obs = copy.deepcopy(obs_dict)
        return obs