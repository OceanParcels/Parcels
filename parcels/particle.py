__all__ = ['Particle']


class Particle(object):
    """Classe encapsualting the basic attributes of a particle"""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "P(%f, %f)" % (self.x, self.y)

    def advect_rk4(self, grid, dt):
        f = dt / 1000. / 1.852 / 60.
        u1, v1 = grid.eval(self.x, self.y)
        x1, y1 = (self.x + u1*.5*f, self.y + v1*.5*f)
        u2, v2 = grid.eval(x1, y1)
        x2, y2 = (self.x + u2*.5*f, self.y + v2*.5*f)
        u3, v3 = grid.eval(x2, y2)
        x3, y3 = (self.x + u3*f, self.y + v3*f)
        u4, v4 = grid.eval(x3, y3)
        self.x += (u1 + 2*u2 + 2*u3 + u4) / 6. * f
        self.y += (v1 + 2*v2 + 2*v3 + v4) / 6. * f
