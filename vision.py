from math import pi, cos, sin, e, sqrt

import numpy as np

_CHANNELS_NONE = [0, 0]
_CHANNELS_WALL = [1, 0]
_CHANNELS_COIN = [0, 1]


class Vision:
    def __init__(self, level, grid_shape,
                 agent_radius=0.45,
                 coin_radius=0.25,
                 signal_count=32,
                 fov=pi / 2,
                 attenuation=0.25):
        self._level = level
        self._grid_shape = grid_shape
        self._agent_radius = agent_radius
        self._signal_count = signal_count
        self._fov = fov
        self._coin_radius = coin_radius
        self._attenuation = attenuation

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
        theta_max = theta_center + (fov / 2)
        theta_step = fov / (total - 1)

        # fan out
        rays = []
        theta = theta_max
        for i in range(0, total):
            point = [near_clip * cos(theta) + origin[0], near_clip * sin(theta) + origin[1]]
            rays.append(Ray(point, theta))
            theta -= theta_step

        return rays

    def _cast(self, ray):

        # there are 2 types of shapes in a level: edges and circles
        edges = self._edges()
        circles = self._circles()

        # select the nearest item the ray intersects
        t_nearest = float('inf')
        channels_nearest = _CHANNELS_NONE

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

        return Signal(ray.point, intersection, self.fog(channels_nearest, t_nearest, self._attenuation))

    def _edges(self):

        # walk between the cells to find the edges

        w = self._grid_shape[1]
        h = self._grid_shape[0]

        grid = self._level.grid
        edges = []

        # walk between rows, draw a line along edges
        for y in range(1, h):
            x0 = None
            for x in range(0, w):
                a = grid[h - y][x]
                b = grid[h - y - 1][x]
                if a != b and x0 is None:
                    x0 = x
                elif a == b and x0 is not None:
                    edges.append(Edge([x0, y], [x, y], _CHANNELS_WALL))
                    x0 = None
            if x0 is not None:
                edges.append(Edge([x0, y], [w, y], _CHANNELS_WALL))

        # walk between columns
        for x in range(1, w):
            y0 = None
            for y in range(0, h):
                a = grid[h - y - 1][x - 1]
                b = grid[h - y - 1][x]
                if a != b and y0 is None:
                    y0 = y
                elif a == b and y0 is not None:
                    edges.append(Edge([x, y0], [x, y], _CHANNELS_WALL))
                    y0 = None
            if y0 is not None:
                edges.append(Edge([x, y0], [x, h], _CHANNELS_WALL))

        return edges

    def _circles(self):

        # coins are circles

        r = self._coin_radius
        c = _CHANNELS_COIN

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

        # represent the ray's points in the circle's local space
        p1 = np.array(ray.point) - np.array(circle.a)
        p2 = np.array(ray.project(1)) - np.array(circle.a)
        r = circle.r

        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]

        x2_m_x1 = x2 - x1
        y2_m_y1 = y2 - y1

        # variables of quadratic formula
        a = x2_m_x1 * x2_m_x1 + y2_m_y1 * y2_m_y1
        b = 2 * (x1 * x2_m_x1 + y1 * y2_m_y1)
        c = x1 * x1 + y1 * y1 - r * r

        delta = b * b - 4 * a * c

        if delta < 0:
            return float('inf')
        else:
            # quadratic formula; only the smaller result
            t = (-b - sqrt(delta)) / (2 * a)
            return t if t >= 0 else float('inf')

    @staticmethod
    def fog(channels, distance, attenuation):
        factor = 1 / pow(e, distance * attenuation)
        return np.array(channels) * [factor]


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
