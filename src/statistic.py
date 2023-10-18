class RunningMean():
    def __init__(self):
        self.mean = 0
        self.n = 0
    
    def update(self,x,n):
        self.mean = (self.mean * self.n + x*n) / (self.n + n)
        self.n += n
    
    def reset(self):
        self.mean = 0
        self.n = 0
    
    def __call__(self):
        return self.mean