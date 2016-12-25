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
            self._coin['position'] = self.coin_positions(level.coins)
        finally:
            self._lock.release()

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
            void main()
            {
                gl_Position = vec4(position, 0.0, 1.0);
                gl_TexCoord[0] = gl_MultiTexCoord0;
            }
        """

        fragment = """
            uniform vec4 circle_color = vec4(0.0, 1.0, 1.0, 1.0);
            uniform vec4 bkg_color = vec4(0.0, 1.0, 1.0, 0.0);
            void main()
            {
                vec2 uv = gl_TexCoord[0].xy;
                float dist = sqrt(dot(uv, uv));
                if (dist < 0.1)
                    gl_FragColor = circle_color;
                else
                    gl_FragColor = bkg_color;
            }
        """

        coin = gloo.Program(vertex, fragment)
        coin['position'] = [0.0] * 4

        return coin

    @staticmethod
    def coin_positions(coins):
        positions = [None] * len(coins) * 4
        offset = 0
        for coin in coins:
            scaled = [coin[0] / 5 - 1, coin[1] / 5 - 1]
            positions[offset + 0] = (scaled[0] - 0.1, scaled[1] - 0.1)
            positions[offset + 1] = (scaled[0] - 0.1, scaled[1] + 0.1)
            positions[offset + 2] = (scaled[0] + 0.1, scaled[1] - 0.1)
            positions[offset + 3] = (scaled[0] + 0.1, scaled[1] + 0.1)
            offset += 4
        return positions;