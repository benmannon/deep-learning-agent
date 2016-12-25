from threading import Lock

import numpy as np
from glumpy import app, gloo, gl


def _grid_program():
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
    _grid['texture'] = np.zeros((10, 10, 1))

    return _grid

_lock = Lock()
_grid = _grid_program()


def init():

    window = app.Window(width=1024, height=1024, aspect=1)

    @window.event
    def on_draw(dt):
        _lock.acquire()
        try:
            window.clear()
            _grid.draw(gl.GL_TRIANGLE_STRIP)
        finally:
            _lock.release()


def update(level):
    _lock.acquire()
    try:
        _grid['texture'] = level.grid
    finally:
        _lock.release()


def run():
    app.run()
