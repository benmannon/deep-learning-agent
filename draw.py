from math import pi
from threading import Lock

import numpy as np
from glumpy import app, gloo, gl


class Draw:
    def __init__(
            self, grid_shape,
            coin_radius=0.05,
            agent_radius=0.09,
            agent_pointer_threshold=pi/4):

        self._lock = Lock()
        self._grid_shape = grid_shape
        self._coin_radius = coin_radius
        self._agent_radius = agent_radius
        self._agent_pointer_threshold = agent_pointer_threshold
        self._grid = self._grid_program()
        self._coin = self._coin_program()
        self._agent = self._agent_program()

    def update(self, level):
        self._lock.acquire()
        try:
            self._grid['texture'] = self.grid_texture(level.grid)
            self._coin['texcoord'] = [(-1, -1), (-1, +1), (+1, +1), (+1, -1)] * len(level.coins)
            self._coin['position'] = self.coin_positions(level.coins)
            self._agent['texcoord'] = [(-1, -1), (-1, +1), (+1, +1), (+1, -1)]
            self._agent['position'] = self.agent_position(level.agent)
            self._agent['theta'] = level.agent.theta
        finally:
            self._lock.release()

    def grid_texture(self, grid):
        return np.reshape(np.repeat(grid, 3), grid.shape + (3,))

    def coin_positions(self, coins):
        positions = [None] * len(coins) * 4
        r = self._coin_radius
        offset = 0
        for coin in coins:
            normal = self.normalize(coin)
            positions[offset + 0] = (normal[0] - r, normal[1] - r)
            positions[offset + 1] = (normal[0] - r, normal[1] + r)
            positions[offset + 2] = (normal[0] + r, normal[1] + r)
            positions[offset + 3] = (normal[0] + r, normal[1] - r)
            offset += 4
        return positions

    def agent_position(self, agent):
        normal = self.normalize(agent.coord)
        r = self._agent_radius
        position = [None] * 4
        position[0] = (normal[0] - r, normal[1] - r)
        position[1] = (normal[0] - r, normal[1] + r)
        position[2] = (normal[0] + r, normal[1] + r)
        position[3] = (normal[0] + r, normal[1] - r)
        return position

    def normalize(self, coord):
        w_half, h_half = self._grid_shape[0] / 2, self._grid_shape[1] / 2
        return [coord[0] / w_half - 1, coord[1] / h_half - 1]

    def run(self):

        window = app.Window(width=1024, height=1024, aspect=1)

        @window.event
        def on_draw(dt):
            self._lock.acquire()
            try:
                window.clear()
                self._grid.draw(gl.GL_TRIANGLE_STRIP)
                self._coin.draw(gl.GL_QUADS)
                self._agent.draw(gl.GL_QUADS)
            finally:
                self._lock.release()

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

        grid = gloo.Program(grid_vertex, grid_fragment, count=4)
        grid['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        grid['texcoord'] = [(0, 1), (0, 0), (1, 1), (1, 0)]
        grid['texture'] = np.zeros(self._grid_shape + (3,))

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
            uniform vec4 bkg_color;
            varying vec2 v_texcoord;
            void main()
            {
                float dist = sqrt(dot(v_texcoord, v_texcoord));
                if (dist < 1)
                    gl_FragColor = circle_color;
                else
                    gl_FragColor = bkg_color;
            }
        """

        coin = gloo.Program(vertex, fragment)
        coin['circle_color'] = [1.0, 0.843, 0.0, 1.0]
        coin['bkg_color'] = [1.0, 0.843, 0.0, 0.0]

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
            uniform vec4 bkg_color;
            uniform float pointer_threshold;
            uniform float theta;
            varying vec2 v_texcoord;
            void main()
            {
                float dist = sqrt(dot(v_texcoord, v_texcoord));
                if (dist < 1)
                {
                    vec2 coord_unit = v_texcoord / dist;
                    float theta_actual = atan(coord_unit.y, coord_unit.x);
                    float theta_diff = abs(theta - theta_actual);
                    if (theta_diff > pointer_threshold)
                        gl_FragColor = circle_color;
                    else
                        gl_FragColor = pointer_color;
                }
                else
                {
                    gl_FragColor = bkg_color;
                }
            }
        """

        agent = gloo.Program(vertex, fragment, count=4)
        agent['circle_color'] = [0.31, 0.89, 0.706, 1.0]
        agent['pointer_color'] = [0.83, 0.98, 0.93, 1.0]
        agent['bkg_color'] = [0.31, 0.89, 0.706, 0.0]
        agent['pointer_threshold'] = self._agent_pointer_threshold

        return agent
