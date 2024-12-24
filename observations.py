import numpy as np
import copy
import cv2

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
        # Convert to BGR for OpenCV
        bgr_img = img[:, :, ::-1]
        _, encoded = cv2.imencode(".jpg", bgr_img)
        return encoded.tobytes()
    
    @property
    def state(self):
        return self._obs['state']
    
    def image(self, cam_idx):
        encoded_image_np = np.frombuffer(self._obs[f'enc_cam_{cam_idx}'], dtype=np.uint8)
        bgr_image = cv2.imdecode(encoded_image_np, cv2.IMREAD_COLOR)
        rgb_image = bgr_image[:, :, ::-1]
        return rgb_image
    
    # Dictionary interface methods
    def keys(self):
        return self._obs.keys()
    
    def items(self):
        return self._obs.items()
    
    def __getitem__(self, key):
        return self._obs[key]
    
    def to_dict(self):
        return copy.deepcopy(self._obs)
    
    @classmethod
    def from_dict(cls, obs_dict):
        obs = cls()
        obs._obs = copy.deepcopy(obs_dict)
        return obs