from math import pi, cos, sin
import numpy as np

class Vision:

    _CHANNELS_NONE = [0, 0]
    _CHANNELS_WALL = [1, 0]
    _CHANNELS_COIN = [0, 1]

    def __init__(self, level, grid_shape, agent_radius=0.45, coin_radius=0.25, signal_count=32, fov=pi / 2):
        self._level = level
        self._grid_shape = grid_shape
        self._agent_radius = agent_radius
        self._signal_count = signal_count
        self._fov = fov
        self._coin_radius = coin_radius

    def look(self):
        signals = []
        for ray in self._rays():
            signals.append(self._cast(ray))
        return signals

    def _rays(self):

        # gather parameters
        origin = self._level.agent.coord
        theta_center = self._level.agent.theta
        near_clip = self._agent_radius
        fov = self._fov
        total = self._signal_count

        # determine the initial angle and step size
        theta_min = theta_center - (fov / 2)
        theta_step = fov / (total - 1)

        # fan out
        rays = []
        theta = theta_min
        for i in range(0, total):
            point = [near_clip * cos(theta) + origin[0], near_clip * sin(theta) + origin[1]]
            rays.append(Ray(point, theta))
            theta += theta_step

        return rays

    def _cast(self, ray):

        # there are 2 types of shapes in a level: edges and circles
        edges = self._edges()
        circles = self._circles()

        # select the nearest item the ray intersects
        t_nearest = float('inf')
        channels_nearest = self._CHANNELS_NONE

        # first check all the edges
        for edge in edges:
            t = self._cast_edge(ray, edge)
            if t < t_nearest:
                t_nearest = t
                channels_nearest = edge.channels

        # next check all the circles
        for circle in circles:
            t = self._cast_circle(ray, circle)
            if t < t_nearest:
                t_nearest = t
                channels_nearest = circle.channels

        intersection = ray.project(t_nearest) if t_nearest != float('inf') else ray.point

        return Signal(ray.point, intersection, channels_nearest)

    def _edges(self):

        # for each grid cell that is not open, there are 4 edges

        w = self._grid_shape[1]
        h = self._grid_shape[0]

        grid = self._level.grid

        # TODO don't return an edge twice for adjacent cells
        edges = []
        for y in range(0, h):
            for x in range(0, w):
                if grid[h - y - 1][x] == 1:
                    edges.extend(self._cell_edges(x, y))

        return edges

    def _cell_edges(self, x, y):

        # all the edges for a single grid cell

        # components
        x0, x1 = x, x + 1
        y0, y1 = y, y + 1

        # vertices
        a = [x0, y0]
        b = [x0, y1]
        c = [x1, y1]
        d = [x1, y0]

        # cells are seen as walls
        ch = self._CHANNELS_WALL

        # construct an edge for each segment
        return [Edge(a, b, ch), Edge(b, c, ch), Edge(c, d, ch), Edge(d, a, ch)]

    def _circles(self):

        # coins are circles

        r = self._coin_radius
        c = self._CHANNELS_COIN

        # just return each coin as a circle
        circles = []
        for coin in self._level.coins:
            circles.append(Circle(coin, r, c))

        return circles

    @staticmethod
    def _cast_edge(ray, edge):

        p = np.array(ray.point)
        r = np.array(ray.project(1.0)) - p
        q = np.array(edge.a)
        s = np.array(edge.b) - q

        t = np.cross(q - p, s) / np.cross(r, s)
        u = np.cross(q - p, r) / np.cross(r, s)

        # return t unless intersection is behind ray or outside edge points
        return t if 0 <= t and 0 <= u <= 1 else float('inf')

    @staticmethod
    def _cast_circle(ray, circle):
        return float('inf')

    @staticmethod
    def _v_subtract(a, b):
        return [a[0] - b[0], a[1] - b[1]]


class Ray:
    def __init__(self, point, theta):
        self.point = point
        self.theta = theta

    def project(self, t):
        x = self.point[0] + t * cos(self.theta)
        y = self.point[1] + t * sin(self.theta)
        return [x, y]


class Signal:
    def __init__(self, a, b, channels):
        self.a = a
        self.b = b
        self.channels = channels


class Edge:
    def __init__(self, a, b, channels):
        self.a = a
        self.b = b
        self.channels = channels

    def __repr__(self):
        return '[a=%s,b=%s,channels=%s]' % (self.a, self.b, self.channels)


class Circle:
    def __init__(self, a, r, channels):
        self.a = a
        self.r = r
        self.channels = channels
