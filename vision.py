from collections import namedtuple
from math import pi, cos, sin, e, sqrt

import numpy as np

_Ray = namedtuple('Ray', 'x y cos sin')
_Signal = namedtuple('Signal', 'ax ay bx by channels')
_Edge = namedtuple('Edge', 'ax ay bx by channels')
_Circle = namedtuple('Circle', 'x y r channels')
_Channel = namedtuple('Channel', 'c0 c1')

_CHANNELS_NONE = _Channel(0, 0)
_CHANNELS_WALL = _Channel(1, 0)
_CHANNELS_COIN = _Channel(0, 1)

CHANNEL_NUM = len(_CHANNELS_NONE)


def _fog(channels, distance, attenuation):
    factor = 1 / pow(e, distance * attenuation)
    return np.array(channels) * [factor]


def _project(ray, t):
    return ray.x + t * ray.cos, ray.y + t * ray.sin


def _cast_circle(ray, circle):

    pr1x, pr1y = ray.x, ray.y
    pcx, pcy = circle.x, circle.y
    r = circle.r

    # exit early if ray is clearly pointing away from circle
    x_dir = ray.cos
    if (x_dir > 0 and pr1x > pcx + r) or (x_dir < 0 and pr1x < pcx - r):
        return float('inf')
    y_dir = ray.sin
    if (y_dir > 0 and pr1y > pcy + r) or (y_dir < 0 and pr1y < pcy - r):
        return float('inf')

    # represent the ray's points in the circle's local space
    pr2x, pr2y = _project(ray, 1)
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

    px, py = ray.x, ray.y
    rx, ry = ray.cos, ray.sin
    qx, qy = edge.ax, edge.ay
    bx, by = edge.bx, edge.by
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
        x = near_clip * cos(theta) + origin[0]
        y = near_clip * sin(theta) + origin[1]
        rays.append(_Ray(x, y, cos(theta), sin(theta)))
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

    bx, by = _project(ray, t_nearest) if t_nearest != float('inf') else [ray.x, ray.y]

    return _Signal(ray.x, ray.y, bx, by, _fog(channels_nearest, t_nearest, attenuation))


def _find_edges(grid):

    # walk between the cells to find the edges

    w = grid.shape[1]
    h = grid.shape[0]

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
                edges.append(_Edge(x0, y, x, y, _CHANNELS_WALL))
                x0 = None
        if x0 is not None:
            edges.append(_Edge(x0, y, w, y, _CHANNELS_WALL))

    # walk between columns
    for x in range(1, w):
        y0 = None
        for y in range(0, h):
            a = grid[h - y - 1][x - 1]
            b = grid[h - y - 1][x]
            if a != b and y0 is None:
                y0 = y
            elif a == b and y0 is not None:
                edges.append(_Edge(x, y0, x, y, _CHANNELS_WALL))
                y0 = None
        if y0 is not None:
            edges.append(_Edge(x, y0, x, h, _CHANNELS_WALL))

    return edges


def _find_circles(coins, coin_radius):

    c = _CHANNELS_COIN

    # just return each coin as a circle
    circles = []
    for coin in coins:
        circles.append(_Circle(coin[0], coin[1], coin_radius, c))

    return circles


class Vision:
    def __init__(self, args, level):
        self._level = level
        self._agent_radius = args.agent_radius
        self._signal_count = args.agent_vision_res
        self._fov = args.agent_vision_fov
        self._attenuation = args.agent_vision_attenuation
        self._coin_radius = args.coin_radius
        self._edges = _find_edges(self._level.grid)

    def look(self):
        signals = []
        agent = self._level.agent

        # there are 2 types of shapes in a level: edges and circles
        edges = self._edges
        circles = _find_circles(self._level.coins, self._coin_radius)

        for ray in _rays(agent.coord, agent.theta, self._agent_radius, self._fov, self._signal_count):
            signals.append(_cast(ray, edges, circles, self._attenuation))

        return signals
