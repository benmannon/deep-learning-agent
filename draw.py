from threading import Lock

import numpy as np
from glumpy import app, gloo, gl


class Draw:
    def __init__(self, grid_shape):
        self._lock = Lock()
        self._grid = self._grid_program(grid_shape)
        self._coin = self._coin_program()

    def update(self, level):
        self._lock.acquire()
        try:
            grid = level.grid
            self._grid['texture'] = np.reshape(np.repeat(grid, 3), grid.shape + (3,))
            self._coin['texcoord'] = [(-1, -1), (-1, +1), (+1, +1), (+1, -1)] * len(level.coins)
            self._coin['position'] = self.coin_positions(level.coins)
        finally:
            self._lock.release()

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

    def run(self):

        window = app.Window(width=1024, height=1024, aspect=1)

        @window.event
        def on_draw(dt):
            self._lock.acquire()
            try:
                window.clear()
                self._grid.draw(gl.GL_TRIANGLE_STRIP)
                self._coin.draw(gl.GL_QUADS)
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

        _grid = gloo.Program(grid_vertex, grid_fragment, count=4)
        _grid['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        _grid['texcoord'] = [(0, 1), (0, 0), (1, 1), (1, 0)]
        _grid['texture'] = np.zeros(shape + (3,))

        return _grid

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
            uniform vec4 circle_color = vec4(1.0, 0.843, 0.0, 1.0);
            uniform vec4 bkg_color = vec4(1.0, 0.843, 0.0, 0.0);
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

        return gloo.Program(vertex, fragment)
