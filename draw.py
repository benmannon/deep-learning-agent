from __future__ import division

from collections import namedtuple
from threading import Lock

import numpy as np
from glumpy import app, gloo, gl


def _calc_level_scale(window_size, grid_shape):
    window_ratio = window_size[1] / window_size[0]
    grid_ratio = grid_shape[0] / grid_shape[1]
    return min(window_ratio / grid_ratio, 0.5)


def _coin_positions(coins, args):
    positions = [None] * len(coins) * 4
    r = args.coin_radius
    offset = 0
    for coin in coins:
        positions[offset + 0] = [coin[0] - r, coin[1] - r]
        positions[offset + 1] = [coin[0] - r, coin[1] + r]
        positions[offset + 2] = [coin[0] + r, coin[1] + r]
        positions[offset + 3] = [coin[0] + r, coin[1] - r]
        offset += 4
    return positions


def _agent_position(agent, args):
    r = args.agent_radius
    position = [None] * 4
    position[0] = [agent.coord[0] - r, agent.coord[1] - r]
    position[1] = [agent.coord[0] - r, agent.coord[1] + r]
    position[2] = [agent.coord[0] + r, agent.coord[1] + r]
    position[3] = [agent.coord[0] + r, agent.coord[1] - r]
    return position


def _line_positions(lines):
    positions = []
    for line in lines:
        positions.append([line.ax, line.ay])
        positions.append([line.bx, line.by])
    return positions


def _update_buffer(program, name, update, use_tuple=False, filler=None):

    buf = program[name]
    unused_len = len(buf) - len(update)

    if unused_len < 0:
        raise ValueError('Vertex buffer overflow')

    # update values
    for i in range(0, len(update)):
        buf[i] = (update[i],) if use_tuple else update[i]

    # pack unused space with filler
    if unused_len > 0:
        buf[len(update):] = [filler] * unused_len


def _grid_texture(grid, args):
    texture = []
    flat_grid = np.reshape(grid, (-1))
    for cell in flat_grid:
        if cell == 1:
            texture.append(args.grid_color + [1.0])
        else:
            texture.append(args.bkg_color + [1.0])
    return np.array(texture)


def _texcoords(repeat=1):
    return [[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]] * repeat


def _line_colors(lines):
    # append each color twice (one per vertex)
    colors = []
    for line in lines:
        rgba = [line.r, line.g, line.b, line.a]
        colors.extend([rgba, rgba])
    return colors


def _grid_program(position, shape):
    grid_vertex = """
        attribute vec2 position;
        attribute vec2 texcoord;
        varying vec2 v_texcoord;
        void main()
        {
            gl_Position = vec4(position, 0.0, 1.0);
            v_texcoord = texcoord;
        }
    """

    grid_fragment = """
        uniform sampler2D texture;
        varying vec2 v_texcoord;
        void main()
        {
            gl_FragColor = texture2D(texture, v_texcoord);
        }
    """

    grid = gloo.Program(grid_vertex, grid_fragment, count=4)
    grid['position'] = position
    grid['texcoord'] = [(0, 1), (0, 0), (1, 1), (1, 0)]
    grid['texture'] = np.zeros(shape + (4,))

    return grid


def _coin_program(coin_color):
    vertex = """
        attribute vec2 position;
        attribute vec2 texcoord;
        varying vec2 v_texcoord;
        void main()
        {
            gl_Position = vec4(position, 0.0, 1.0);
            v_texcoord = texcoord;
        }
    """

    fragment = """
        uniform vec4 circle_color;
        uniform vec4 border_color;
        uniform vec4 bkg_color;
        varying vec2 v_texcoord;
        void main()
        {
            float dist = sqrt(dot(v_texcoord, v_texcoord));
            if (dist < 0.9)
                // inside the border
                if (abs(v_texcoord.x) < 0.1 && abs(v_texcoord.y) < 0.5)
                    // draw a vertical slot
                    gl_FragColor = border_color;
                else
                    gl_FragColor = circle_color;
            else if (dist < 1)
                // the border
                gl_FragColor = border_color;
            else
                // outside the border
                gl_FragColor = bkg_color;
        }
    """

    coin = gloo.Program(vertex, fragment)
    coin['circle_color'] = coin_color + [1.0]
    coin['border_color'] = [0.0, 0.0, 0.0, 1.0]
    coin['bkg_color'] = coin_color + [0.0]

    return coin


def _agent_program(agent_color, pointer_brightness, pointer_size):
    vertex = """
        attribute vec2 position;
        attribute vec2 texcoord;
        varying vec2 v_texcoord;
        void main()
        {
            gl_Position = vec4(position, 0.0, 1.0);
            v_texcoord = texcoord;
        }
    """

    fragment = """
        uniform vec4 circle_color;
        uniform vec4 pointer_color;
        uniform vec4 border_color;
        uniform vec4 bkg_color;
        uniform float pointer_threshold;
        uniform float theta;
        varying vec2 v_texcoord;
        void main()
        {
            float dist = sqrt(dot(v_texcoord, v_texcoord));
            if (dist < 0.9)
            {
                // inside the border; calculate angle to draw pointer
                vec2 coord_unit = v_texcoord / dist;
                float theta_actual = atan(coord_unit.y, coord_unit.x);
                float theta_diff = theta - theta_actual;
                float bounded_diff = abs(atan(sin(theta_diff), cos(theta_diff)));
                if (bounded_diff > pointer_threshold)
                    // outside the pointer arc
                    gl_FragColor = circle_color;
                else
                    // inside the pointer arc
                    gl_FragColor = pointer_color;
            }
            else if (dist < 1)
            {
                // the border
                gl_FragColor = border_color;
            }
            else
            {
                // outside the border
                gl_FragColor = bkg_color;
            }
        }
    """

    # pointer color is a brighter shade of the agent color
    b = pointer_brightness
    pointer_r = b + agent_color[0] - (b * agent_color[0])
    pointer_g = b + agent_color[1] - (b * agent_color[1])
    pointer_b = b + agent_color[2] - (b * agent_color[2])
    pointer_color = [pointer_r, pointer_g, pointer_b]

    agent = gloo.Program(vertex, fragment, count=4)
    agent['circle_color'] = agent_color + [1.0]
    agent['pointer_color'] = pointer_color + [1.0]
    agent['border_color'] = [0.0, 0.0, 0.0, 1.0]
    agent['bkg_color'] = agent_color + [0.0]
    agent['pointer_threshold'] = pointer_size / 2

    return agent


def _lines_program():
    vertex = """
        attribute vec2 position;
        attribute vec4 line_color;
        varying vec4 v_line_color;
        void main()
        {
            gl_Position = vec4(position, 0.0, 1.0);
            v_line_color = line_color;
        }
    """

    fragment = """
        varying vec4 v_line_color;
        void main()
        {
            gl_FragColor = v_line_color;
        }
    """

    return gloo.Program(vertex, fragment)


def _sight_program(shape):
    grid_vertex = """
        attribute vec2 position;
        attribute vec2 texcoord;
        varying vec2 v_texcoord;
        void main()
        {
            gl_Position = vec4(position, 0.0, 1.0);
            v_texcoord = texcoord;
        }
    """

    grid_fragment = """
        uniform sampler2D texture;
        varying vec2 v_texcoord;
        void main()
        {
            gl_FragColor = texture2D(texture, v_texcoord);
        }
    """

    sight = gloo.Program(grid_vertex, grid_fragment, count=4)
    sight['position'] = [(0, -0.25), (0, 0.25), (1, -0.25), (1, 0.25)]
    sight['texcoord'] = [(0, 1), (0, 0), (1, 1), (1, 0)]
    sight['texture'] = np.zeros(shape + (4,))

    return sight


class Draw:
    def __init__(self, args, grid_shape):

        self._lock = Lock()
        self._args = args
        self._window_height = int(args.window_width / 2)
        self._level_scale = _calc_level_scale([args.window_width, self._window_height], grid_shape)
        self._grid_shape = grid_shape
        self._agent_pointer_threshold = args.agent_vision_fov / 2

        grid_w = self._grid_shape[1]
        grid_h = self._grid_shape[0]
        grid_pos = self._normalize_each([(0, 0), (0, grid_h), (grid_w, 0), (grid_w, grid_h)])

        self._grid = _grid_program(grid_pos, grid_shape)
        self._coin = _coin_program(args.coin_color)
        self._agent = _agent_program(args.agent_color, args.agent_pointer_brightness, args.agent_vision_fov)
        self._lines = _lines_program()
        self._sight = _sight_program((1, args.agent_vision_res))
        self._initialized = False

    def update(self, level, lines, sight_colors):
        self._lock.acquire()
        try:
            self._grid['texture'] = _grid_texture(level.grid, self._args)
            self._sight['texture'] = np.array(sight_colors)
            self._agent['theta'] = level.agent.theta
            if self._initialized:
                _update_buffer(self._coin, 'texcoord', _texcoords(len(level.coins)), use_tuple=True, filler=([0, 0],))
                _update_buffer(self._coin, 'position', self._normalize_each(_coin_positions(level.coins, self._args)), use_tuple=True, filler=([0, 0],))
                _update_buffer(self._agent, 'position', self._normalize_each(_agent_position(level.agent, self._args)))
                _update_buffer(self._lines, 'position', self._normalize_each(_line_positions(lines)), use_tuple=True, filler=([0, 0],))
                _update_buffer(self._lines, 'line_color', _line_colors(lines), use_tuple=True, filler=([0, 0, 0, 0],))
            else:
                self._coin['texcoord'] = _texcoords(len(level.coins))
                self._coin['position'] = self._normalize_each(_coin_positions(level.coins, self._args))
                self._agent['texcoord'] = [(-1, -1), (-1, +1), (+1, +1), (+1, -1)]
                self._agent['position'] = self._normalize_each(_agent_position(level.agent, self._args))
                self._lines['position'] = self._normalize_each(_line_positions(lines))
                self._lines['line_color'] = _line_colors(lines)
                self._initialized = True
        finally:
            self._lock.release()

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
            self._lock.acquire()
            try:
                window.clear()
                self._grid.draw(gl.GL_TRIANGLE_STRIP)
                self._coin.draw(gl.GL_QUADS)
                self._agent.draw(gl.GL_QUADS)
                self._lines.draw(gl.GL_LINES)
                self._sight.draw(gl.GL_TRIANGLE_STRIP)
            finally:
                self._lock.release()

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
