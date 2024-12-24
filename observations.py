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
    def __init__(self, obs_dict=None):
        if obs_dict is None:
            self._obs = {
                'state': np.zeros(7, dtype=np.float32),
                'enc_cam_0': self._create_encoded_image()
            }
        else:
            self._obs = obs_dict
        self.prev = None
    
    def _create_encoded_image(self):
        x = np.arange(256)[:, None]
        y = np.arange(256)[None, :]
        checker = ((x + y) % 32 < 16).astype(np.uint8) * 255
        img = np.stack([checker] * 3, axis=-1)
        bgr_img = img[:, :, ::-1]
        _, encoded = cv2.imencode(".jpg", bgr_img)
        return encoded.tobytes()
    
    def __getstate__(self):
        """Tell pickle what to save"""
        return {'_obs': self._obs, 'prev': self.prev}
    
    def __setstate__(self, state):
        """Tell pickle how to restore"""
        self._obs = state['_obs']
        self.prev = state['prev']
    
    @property
    def state(self):
        return self._obs['state']
    
    def image(self, cam_idx):
        encoded_image_np = np.frombuffer(self._obs[f'enc_cam_{cam_idx}'], dtype=np.uint8)
        bgr_image = cv2.imdecode(encoded_image_np, cv2.IMREAD_COLOR)
        rgb_image = bgr_image[:, :, ::-1]
        return rgb_image
    
    def keys(self):
        return self._obs.keys()
    
    def items(self):
        return self._obs.items()
    
    def __getitem__(self, key):
        return self._obs[key]
    
    def to_dict(self):
        return copy.deepcopy(self._obs)