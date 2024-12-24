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
        self.state = np.zeros(7, dtype=np.float32)
        self._image = DummyImage()
        self.prev = None
        self._init_obs()
    
    def _init_obs(self):
        """Initialize the observation dictionary"""
        self._obs = {
            'state': self.state,
            'enc_cam_0': self._create_encoded_image()
        }
    
    def _create_encoded_image(self):
        """Create a checkerboard pattern and encode it"""
        x = np.arange(256)[:, None]
        y = np.arange(256)[None, :]
        checker = ((x + y) % 32 < 16).astype(np.uint8) * 255
        img = np.stack([checker] * 3, axis=-1)
        return img.tobytes()
    
    def __getstate__(self):
        """Called when pickling"""
        return {
            'state': self.state,
            'prev': self.prev,
            '_image': self._image
        }
    
    def __setstate__(self, state_dict):
        """Called when unpickling"""
        self.state = state_dict['state']
        self.prev = state_dict['prev']
        self._image = state_dict['_image']
        self._init_obs()
    
    def image(self, cam_idx):
        """Return RGB image of shape (H, W, 3)"""
        return self._image.image(cam_idx)
    
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