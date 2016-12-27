from __future__ import division

from math import pi
from threading import Lock

import numpy as np
from glumpy import app, gloo, gl


class Draw:
    def __init__(
            self, grid_shape,
            window_size=[1024, 1024],
            level_scale=0.5,
            coin_radius=0.25,
            agent_radius=0.45,
            agent_pointer_threshold=pi / 4,
            grid_color=[0.4, 0.4, 0.4],
            coin_color=[1.0, 0.843, 0.0],
            agent_color=[0.15, 0.45, 0.35],
            agent_pointer_brightness=0.5,
            bkg_color=[1.0, 1.0, 1.0]):

        self._lock = Lock()
        self._window_size = window_size
        self._level_scale = level_scale
        self._grid_shape = grid_shape
        self._coin_radius = coin_radius
        self._agent_radius = agent_radius
        self._agent_pointer_threshold = agent_pointer_threshold
        self._grid_color = grid_color
        self._coin_color = coin_color
        self._agent_color = agent_color
        self._agent_pointer_brightness = agent_pointer_brightness
        self._bkg_color = bkg_color
        self._grid = self._grid_program()
        self._coin = self._coin_program()
        self._agent = self._agent_program()
        self._lines = self._lines_program()
        self._initialized = False

    def update(self, level, lines):
        self._lock.acquire()
        try:
            self._grid['texture'] = self.grid_texture(level.grid)
            if self._initialized:
                self._update_buffer(self._coin, 'texcoord', self.texcoords(len(level.coins)), use_tuple=True)
                self._update_buffer(self._coin, 'position', self.coin_positions(level.coins), use_tuple=True)
                self._update_buffer(self._agent, 'position', self.agent_position(level.agent))
                self._agent['theta'] = level.agent.theta
                self._update_buffer(self._lines, 'position', self.line_positions(lines), use_tuple=True)
                self._update_buffer(self._lines, 'line_color', self.line_colors(lines), use_tuple=True)
            else:
                self._coin['texcoord'] = self.texcoords(len(level.coins))
                self._coin['position'] = self.coin_positions(level.coins)
                self._agent['texcoord'] = [(-1, -1), (-1, +1), (+1, +1), (+1, -1)]
                self._agent['position'] = self.agent_position(level.agent)
                self._agent['theta'] = level.agent.theta
                self._lines['position'] = self.line_positions(lines)
                self._lines['line_color'] = self.line_colors(lines)
                self._initialized = True
        finally:
            self._lock.release()

    def _update_buffer(self, program, name, update, use_tuple=False):

        buf = program[name]

        # resize if necessary; can only get smaller
        if len(update) < len(buf):
            buf = buf[0:len(update)]

        # update values
        for i in range(0, len(buf)):
            buf[i] = (update[i],) if use_tuple else update[i]

        program[name] = buf

    def grid_texture(self, grid):
        texture = []
        flat_grid = np.reshape(grid, (-1))
        for cell in flat_grid:
            if (cell == 1):
                texture.append(self._grid_color + [1.0])
            else:
                texture.append(self._bkg_color + [1.0])
        return np.array(texture)

    def texcoords(self, repeat=1):
        return [[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]] * repeat

    def coin_positions(self, coins):
        positions = [None] * len(coins) * 4
        r = self._coin_radius
        offset = 0
        for coin in coins:
            positions[offset + 0] = self.normalize([coin[0] - r, coin[1] - r])
            positions[offset + 1] = self.normalize([coin[0] - r, coin[1] + r])
            positions[offset + 2] = self.normalize([coin[0] + r, coin[1] + r])
            positions[offset + 3] = self.normalize([coin[0] + r, coin[1] - r])
            offset += 4
        return positions

    def agent_position(self, agent):
        r = self._agent_radius
        position = [None] * 4
        position[0] = self.normalize([agent.coord[0] - r, agent.coord[1] - r])
        position[1] = self.normalize([agent.coord[0] - r, agent.coord[1] + r])
        position[2] = self.normalize([agent.coord[0] + r, agent.coord[1] + r])
        position[3] = self.normalize([agent.coord[0] + r, agent.coord[1] - r])
        return position

    def line_positions(self, lines):
        positions = []
        for line in lines:
            positions.append(self.normalize(line.a))
            positions.append(self.normalize(line.b))
        return positions

    def line_colors(self, lines):
        # append each color twice (one per vertex)
        colors = []
        for line in lines:
            colors.append(line.color + [1.0])
            colors.append(line.color + [1.0])
        return colors

    def run(self, key_handler=None, close_handler=None):

        config = app.configuration.Configuration()
        config.samples = 8

        window_w = self._window_size[0]
        window_h = self._window_size[1]
        window = app.Window(width=window_w, height=window_h, title="Agent Training", config=config)

        @window.event
        def on_draw(dt):
            self._lock.acquire()
            try:
                window.clear()
                self._grid.draw(gl.GL_TRIANGLE_STRIP)
                self._coin.draw(gl.GL_QUADS)
                self._agent.draw(gl.GL_QUADS)
                self._lines.draw(gl.GL_LINES)
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

    def _grid_program(self):
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

        w = self._grid_shape[1]
        h = self._grid_shape[0]

        grid = gloo.Program(grid_vertex, grid_fragment, count=4)
        grid['position'] = self.normalize_each([(0, 0), (0, h), (w, 0), (w, h)])
        grid['texcoord'] = [(0, 1), (0, 0), (1, 1), (1, 0)]
        grid['texture'] = np.zeros(self._grid_shape + (4,))

        return grid

    def _coin_program(self):
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
        coin['circle_color'] = self._coin_color + [1.0]
        coin['border_color'] = [0.0, 0.0, 0.0, 1.0]
        coin['bkg_color'] = self._coin_color + [0.0]

        return coin

    def _agent_program(self):
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
        b = self._agent_pointer_brightness
        pointer_r = b + self._agent_color[0] - (b * self._agent_color[0])
        pointer_g = b + self._agent_color[1] - (b * self._agent_color[1])
        pointer_b = b + self._agent_color[2] - (b * self._agent_color[2])
        pointer_color = [pointer_r, pointer_g, pointer_b]

        agent = gloo.Program(vertex, fragment, count=4)
        agent['circle_color'] = self._agent_color + [1.0]
        agent['pointer_color'] = pointer_color + [1.0]
        agent['border_color'] = [0.0, 0.0, 0.0, 1.0]
        agent['bkg_color'] = self._agent_color + [0.0]
        agent['pointer_threshold'] = self._agent_pointer_threshold

        return agent

    def _lines_program(self):
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

    def normalize_each(self, coords):
        normals = []
        for coord in coords:
            normals.append(self.normalize(coord))
        return normals

    def normalize(self, coord):

        # position level in top-left corner of screen
        #
        # scale determines the amount of horizontal space that is covered
        # vertical space is used as needed, maintaining proper aspect ratio

        grid_aspect = self._grid_shape[0] / self._grid_shape[1]
        window_aspect = self._window_size[0] / self._window_size[1]

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


class Line:
    def __init__(self, a=[0.0, 0.0], b=[1.0, 0.0], color=[0.5, 0.5, 0.5]):
        self.a = a
        self.b = b
        self.color = color
