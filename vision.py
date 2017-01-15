from math import pi, cos, sin, e, sqrt

import numpy as np

CHANNEL_NUM = 2

_CHANNELS_NONE = [0, 0]
_CHANNELS_WALL = [1, 0]
_CHANNELS_COIN = [0, 1]


def _fog(channels, distance, attenuation):
    factor = 1 / pow(e, distance * attenuation)
    return np.array(channels) * [factor]


def _cast_circle(ray, circle):

    pr1x, pr1y = ray.point
    pcx, pcy = circle.a
    r = circle.r

    # exit early if ray is clearly pointing away from circle
    x_dir = ray.cos_theta()
    if (x_dir > 0 and pr1x > pcx + r) or (x_dir < 0 and pr1x < pcx - r):
        return float('inf')
    y_dir = ray.sin_theta()
    if (y_dir > 0 and pr1y > pcy + r) or (y_dir < 0 and pr1y < pcy - r):
        return float('inf')

    # represent the ray's points in the circle's local space
    pr2x, pr2y = ray.project(1)
    x1, y1 = pr1x - pcx, pr1y - pcy
    x2, y2 = pr2x - pcx, pr2y - pcy

    # reuse these calculations
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


def _cast_edge(ray, edge):

    px, py = ray.point
    rx, ry = ray.cos_theta(), ray.sin_theta()
    qx, qy = edge.a
    bx, by = edge.b
    sx, sy = bx - qx, by - qy

    q_m_p = [qx - px, qy - py]
    cross_r_s = _cross_2d([rx, ry], [sx, sy])
    if cross_r_s == 0:
        return float('inf')

    t = _cross_2d(q_m_p, [sx, sy]) / cross_r_s
    u = _cross_2d(q_m_p, [rx, ry]) / cross_r_s

    # return t unless intersection is behind ray or outside edge points
    return t if 0 <= t and 0 <= u <= 1 else float('inf')


def _cross_2d(v1, v2):
    x1, y1 = v1[0], v1[1]
    x2, y2 = v2[0], v2[1]
    return x1 * y2 - x2 * y1


def _rays(origin, theta_center, near_clip, fov, total):

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


def _cast(ray, edges, circles, attenuation):

    # select the nearest item the ray intersects
    t_nearest = float('inf')
    channels_nearest = _CHANNELS_NONE

    # first check all the edges
    for edge in edges:
        t = _cast_edge(ray, edge)
        if t < t_nearest:
            t_nearest = t
            channels_nearest = edge.channels

    # next check all the circles
    for circle in circles:
        t = _cast_circle(ray, circle)
        if t < t_nearest:
            t_nearest = t
            channels_nearest = circle.channels

    intersection = ray.project(t_nearest) if t_nearest != float('inf') else ray.point

    return Signal(ray.point, intersection, _fog(channels_nearest, t_nearest, attenuation))


def _find_edges(grid, shape):

    # walk between the cells to find the edges

    w = shape[1]
    h = shape[0]

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


def _find_circles(coins, coin_radius):

    c = _CHANNELS_COIN

    # just return each coin as a circle
    circles = []
    for coin in coins:
        circles.append(Circle(coin, coin_radius, c))

    return circles


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
        self._edges = _find_edges(self._level.grid, self._grid_shape)

    def look(self):
        signals = []
        agent = self._level.agent

        # there are 2 types of shapes in a level: edges and circles
        edges = self._edges
        circles = _find_circles(self._level.coins, self._coin_radius)

        for ray in _rays(agent.coord, agent.theta, self._agent_radius, self._fov, self._signal_count):
            signals.append(_cast(ray, edges, circles, self._attenuation))

        return signals


class Ray:
    def __init__(self, point, theta):
        self.point = point
        self.theta = theta
        self._cos = None
        self._sin = None

    def project(self, t):
        x = self.point[0] + t * self.cos_theta()
        y = self.point[1] + t * self.sin_theta()
        return [x, y]

    def cos_theta(self):
        if self._cos is None:
            self._cos = cos(self.theta)
        return self._cos

    def sin_theta(self):
        if self._sin is None:
            self._sin = sin(self.theta)
        return self._sin


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
