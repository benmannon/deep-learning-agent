from threading import Lock

import numpy as np
from glumpy import app, gloo, gl


class Draw:
    def __init__(self, grid_shape):
        self._lock = Lock()
        self._grid = self._grid_program(grid_shape)

    def update(self, level):
        self._lock.acquire()
        try:
            grid = level.grid
            self._grid['texture'] = np.reshape(np.repeat(grid, 3), grid.shape + (3,))
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
