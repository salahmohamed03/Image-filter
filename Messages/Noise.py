

class Noise:
    def __init__(self, noise = 'salt and pepper', filter = 'average', saltRatio = 0, pepperRatio = 0):
        self.noise = noise
        self.filter = filter
        self.saltRatio = saltRatio
        self.pepperRatio = pepperRatio