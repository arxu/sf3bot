from gym.spaces.space import Space
import numpy as np

class Incremental(Space):
    def __init__(self, start, stop, num, **kwargs):
        self.values = np.linspace(start, stop, num, **kwargs)
        super().__init__(self.values.shape, self.values.dtype)
        
    def sample(self):
        return np.random.choice(self.values)
    
    def contains(self,x ):
        return x in self.values