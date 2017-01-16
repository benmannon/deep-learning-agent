from __future__ import division

from collections import namedtuple
from threading import Lock

import numpy as np
from glumpy import app, gloo, gl

import shaders


def _calc_level_scale(window_size, grid_shape):
    window_ratio = window_size[1] / window_size[0]
    grid_ratio = grid_shape[0] / grid_shape[1]
    return min(window_ratio / grid_ratio, 0.5)


def _circle_square(point, r):
    x, y = point
    return [(x - r, y - r),
            (x - r, y + r),
            (x + r, y + r),
            (x + r, y - r)]


def _circle_squares(points, r):
    positions = [None] * len(points) * 4
    offset = 0
    for point in points:
        positions[offset:offset + 4] = _circle_square(point, r)
        offset += 4
    return positions


def _update_buffer(buf, update, use_tuple=False):
    for i in range(0, len(update)):
        buf[i] = (update[i],) if use_tuple else update[i]


def _grid_texture(grid, args):
    texture = []
    flat_grid = np.reshape(grid, (-1))
    for cell in flat_grid:
        if cell == 1:
            texture.append(args.grid_color + [1.0])
        else:
            texture.append(args.bkg_color + [1.0])
    return np.array(texture)


def _grid_program(position, shape):
    grid_vertex = shaders.IMAGE.vertex
    grid_fragment = shaders.IMAGE.fragment

    grid = gloo.Program(grid_vertex, grid_fragment, count=4)
    grid['position'] = position
    grid['texcoord'] = [(0, 1), (0, 0), (1, 1), (1, 0)]
    grid['texture'] = np.zeros(shape + (4,))

    return grid


def _coin_program(coin_color):
    vertex = shaders.COIN.vertex
    fragment = shaders.COIN.fragment

    coin = gloo.Program(vertex, fragment)
    coin['circle_color'] = coin_color + [1.0]
    coin['border_color'] = [0.0, 0.0, 0.0, 1.0]
    coin['bkg_color'] = coin_color + [0.0]

    return coin


def _agent_program(agent_color, pointer_brightness, pointer_size):
    vertex = shaders.AGENT.vertex
    fragment = shaders.AGENT.fragment

    # pointer color is a brighter shade of the agent color
    b = pointer_brightness
    pointer_r = b + agent_color[0] - (b * agent_color[0])
    pointer_g = b + agent_color[1] - (b * agent_color[1])
    pointer_b = b + agent_color[2] - (b * agent_color[2])
    pointer_color = [pointer_r, pointer_g, pointer_b]

    agent = gloo.Program(vertex, fragment, count=4)
    agent['position'] = [(0, 0), (0, 0), (0, 0), (0, 0)]
    agent['texcoord'] = [(-1, -1), (-1, +1), (+1, +1), (+1, -1)]
    agent['circle_color'] = agent_color + [1.0]
    agent['pointer_color'] = pointer_color + [1.0]
    agent['border_color'] = [0.0, 0.0, 0.0, 1.0]
    agent['bkg_color'] = agent_color + [0.0]
    agent['pointer_threshold'] = pointer_size / 2

    return agent


def _line_program():
    vertex = shaders.LINE.vertex
    fragment = shaders.LINE.fragment

    return gloo.Program(vertex, fragment)


def _sight_program(shape):
    grid_vertex = shaders.IMAGE.vertex
    grid_fragment = shaders.IMAGE.fragment

    sight = gloo.Program(grid_vertex, grid_fragment, count=4)
    sight['position'] = [(0, -0.25), (0, 0.25), (1, -0.25), (1, 0.25)]
    sight['texcoord'] = [(0, 1), (0, 0), (1, 1), (1, 0)]
    sight['texture'] = np.zeros(shape + (4,))

    return sight


def _update_agent(agent_program, coord, theta, r, transform):
    _update_buffer(agent_program['position'], transform(_circle_square(coord, r)))
    agent_program['theta'] = theta


def _update_coins(coin_programs, coins, coin_color, r, transform):

    n_coins = len(coins)

    # update existing coins
    for i in range(0, min(len(coin_programs), n_coins)):
        _update_buffer(coin_programs[i]['position'], transform(_circle_square(coins[i], r)), use_tuple=True)

    # create new coins if necessary
    for i in range(min(len(coin_programs), n_coins), n_coins):
        program = _coin_program(coin_color)
        program['texcoord'] = [(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)]
        program['position'] = transform(_circle_square(coins[i], r))
        coin_programs.append(program)

    # delete coins if necessary
    for i in range(n_coins, len(coin_programs)):
        coin_programs[i].delete()
    del coin_programs[n_coins:]


def _update_lines(line_programs, lines, transform):

    n_lines = len(lines)

    # update existing lines
    for i in range(0, min(len(line_programs), n_lines)):
        ax, ay, bx, by, r, g, b, a = lines[i]
        _update_buffer(line_programs[i]['position'], transform([(ax, ay), (bx, by)]), use_tuple=True)
        _update_buffer(line_programs[i]['line_color'], [(r, g, b, a), (r, g, b, a)], use_tuple=True)

    # create new lines if necessary
    for i in range(min(len(line_programs), n_lines), n_lines):
        ax, ay, bx, by, r, g, b, a = lines[i]
        program = _line_program()
        program['position'] = transform([(ax, ay), (bx, by)])
        program['line_color'] = [(r, g, b, a), (r, g, b, a)]
        line_programs.append(program)

    # delete lines if necessary
    for i in range(n_lines, len(line_programs)):
        lines[i].delete()
    del lines[n_lines:]


class Draw:
    def __init__(self, args, grid_shape, update_hook):

        self._args = args
        self._window_height = int(args.window_width / 2)
        self._level_scale = _calc_level_scale([args.window_width, self._window_height], grid_shape)
        self._grid_shape = grid_shape
        self._agent_pointer_threshold = args.agent_vision_fov / 2

        grid_w = self._grid_shape[1]
        grid_h = self._grid_shape[0]
        grid_pos = self._normalize_each([(0, 0), (0, grid_h), (grid_w, 0), (grid_w, grid_h)])

        self._grid = _grid_program(grid_pos, grid_shape)
        self._coins = []
        self._agent = _agent_program(args.agent_color, args.agent_pointer_brightness, args.agent_vision_fov)
        self._lines = []
        self._sight = _sight_program((1, args.agent_vision_res))

        self._update_hook = update_hook

    def _update(self, grid, agent_coord, agent_theta, coins, lines, sight_colors):

        self._grid['texture'] = _grid_texture(grid, self._args)
        self._sight['texture'] = np.array(sight_colors)

        _update_agent(self._agent, agent_coord, agent_theta, self._args.agent_radius, self._normalize_each)
        _update_coins(self._coins, coins, self._args.coin_color, self._args.coin_radius, self._normalize_each)
        _update_lines(self._lines, lines, self._normalize_each)

    def _normalize_each(self, coords):
        normals = []
        for coord in coords:
            normals.append(self._normalize(coord))
        return normals

    def _normalize(self, coord):

        # position level in top-left corner of screen
        #
        # scale determines the amount of horizontal space that is covered
        # vertical space is used as needed, maintaining proper aspect ratio

        grid_aspect = self._grid_shape[0] / self._grid_shape[1]
        window_aspect = self._args.window_width / self._window_height

        xmin = -1
        xmax = -1 + (2 * self._level_scale)
        ymin = 1 - (2 * self._level_scale) * grid_aspect * window_aspect
        ymax = 1

        gw = self._grid_shape[1]
        gh = self._grid_shape[0]
        x_unit = coord[0] / gw
        y_unit = coord[1] / gh
        x = (xmax - xmin) * x_unit + xmin
        y = (ymax - ymin) * y_unit + ymin

        return [x, y]

    def run(self, key_handler=None, close_handler=None):

        config = app.configuration.Configuration()
        config.samples = 8

        window_w = self._args.window_width
        window_h = self._window_height
        window = app.Window(width=window_w, height=window_h, title="Simulator", config=config)

        @window.event
        def on_draw(dt):

            self._update(*self._update_hook())

            window.clear()
            self._grid.draw(gl.GL_TRIANGLE_STRIP)
            for coin in self._coins:
                coin.draw(gl.GL_QUADS)
            self._agent.draw(gl.GL_QUADS)
            for line in self._lines:
                line.draw(gl.GL_LINES)
            self._sight.draw(gl.GL_TRIANGLE_STRIP)
        if key_handler is not None:
            @window.event
            def on_key_press(symbol, modifiers):
                key_handler(symbol, modifiers)

        if close_handler is not None:
            @window.event
            def on_close():
                close_handler()

        app.run()


Line = namedtuple('Line', 'ax ay bx by r g b a')
