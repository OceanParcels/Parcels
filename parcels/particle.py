__all__ = ['Particle']


class Particle(object):
    """Classe encapsualting the basic attributes of a particle"""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "P(%f, %f)" % (self.x, self.y)
