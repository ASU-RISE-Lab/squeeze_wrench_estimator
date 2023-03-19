import numpy as np

class Median_LPF:
    def __init__(self):
        self.lp_vase1 = [0, 0]
        self.lp_vase2 = [0, 0]

    def med_lp(self, v_window, lp_alpha):
        med_vase = []
        med_filter = []

        for j in range(len(v_window)):
            h = j + 1
            med_vase.append(v_window[len(v_window)-h])
        
        med_filter = np.median(med_vase)
        self.lp_vase1[0] = lp_alpha*med_filter + (1-lp_alpha)*self.lp_vase1[1]
        self.lp_vase2[0] = lp_alpha*self.lp_vase1[0] + (1-lp_alpha)*self.lp_vase2[1]
        lp_filter = self.lp_vase2[0]
        self.lp_vase1[1] = self.lp_vase1[0]
        self.lp_vase2[1] = self.lp_vase2[0]
        return lp_filter