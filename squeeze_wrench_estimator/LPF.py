

def lp_filter(self,current, previous, alpha):
    return alpha*current + (1-alpha)*previous