from math import pi
from threading import Lock

import numpy as np
from glumpy import app, gloo, gl


class Draw:
    def __init__(self, grid_shape):
        self._lock = Lock()
        self._grid = self._grid_program(grid_shape)
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

    @staticmethod
    def coin_positions(coins):
        positions = [None] * len(coins) * 4
        offset = 0
        for coin in coins:
            scaled = [coin[0] / 5 - 1, coin[1] / 5 - 1]
            positions[offset + 0] = (scaled[0] - 0.05, scaled[1] - 0.05)
            positions[offset + 1] = (scaled[0] - 0.05, scaled[1] + 0.05)
            positions[offset + 2] = (scaled[0] + 0.05, scaled[1] + 0.05)
            positions[offset + 3] = (scaled[0] + 0.05, scaled[1] - 0.05)
            offset += 4
        return positions;

    @staticmethod
    def agent_position(agent):
        scaled = [agent.x / 5 - 1, agent.y / 5 - 1]
        position = [None] * 4
        position[0] = (scaled[0] - 0.09, scaled[1] - 0.09)
        position[1] = (scaled[0] - 0.09, scaled[1] + 0.09)
        position[2] = (scaled[0] + 0.09, scaled[1] + 0.09)
        position[3] = (scaled[0] + 0.09, scaled[1] - 0.09)
        return position;

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

    @staticmethod
    def _grid_program(shape):
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
        grid['texture'] = np.zeros(shape + (3,))

        return grid

    @staticmethod
    def _coin_program():
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

    @staticmethod
    def _agent_program():
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
        agent['pointer_threshold'] = pi / 4

        return agent
