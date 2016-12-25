from glumpy import app, gloo, gl
import numpy as np
from threading import Lock

_lock = Lock()
_program_grid = None


def init():
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
        uniform sampler2D texture
        varying vec2 v_texcoord;
        void main()
        {
            gl_FragColor = texture2D(texture, v_texcoord)
        }
    """

    _program_grid = gloo.Program(grid_vertex, grid_fragment, count=4)
    _program_grid['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
    _program_grid['texcoord'] = [(0, 1), (0, 0), (1, 1), (1, 0)]
    _program_grid['texture'] = np.zeros((1, 1, 3))

    window = app.Window(width=1024, height=1024, aspect=1)

    @window.event
    def on_draw(dt):
        _lock.acquire()
        try:
            window.clear()
            _program_grid.draw(gl.GL_TRIANGLE_STRIP)
        finally:
            _lock.release()


def update(grid):
    _program_grid['texture'] = np.reshape(np.repeat(grid, 3), (-1, -1, 3))


def run():
    app.run()
