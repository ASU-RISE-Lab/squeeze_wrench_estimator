import numpy as np

class Moving_Avg_Filter:
    def __init__(self,size=10) -> None:
        self.buffer = np.zeros(size)

    def callback(self,value):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = value
        return np.mean(self.buffer)