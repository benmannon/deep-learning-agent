from __future__ import division

import threading

import controller
import level
from draw import Draw, Line
from vision import Vision


class Simulator:
    def __init__(self, agent_vision_res, agent_vision_fov, agent_vision_attenuation, agent_radius, agent_stride,
                 agent_turn, coin_radius, window_width, grid_color, coin_color, agent_color, agent_pointer_brightness,
                 bkg_color):

        self._lvl = level.square()
        self._grid_color = grid_color
        self._coin_color = coin_color

        self._draw = Draw(self._lvl.grid.shape,
                          sight_res=agent_vision_res,
                          window_width=window_width,
                          coin_radius=coin_radius,
                          agent_radius=agent_radius,
                          grid_color=grid_color,
                          coin_color=coin_color,
                          agent_color=agent_color,
                          agent_pointer_brightness=agent_pointer_brightness,
                          bkg_color=bkg_color)

        self._ctrl = controller.Controller(self._lvl,
                                           agent_stride=agent_stride,
                                           agent_turn=agent_turn,
                                           agent_radius=agent_radius,
                                           coin_radius=coin_radius)

        self._vision = Vision(self._lvl, self._lvl.grid.shape,
                              agent_radius=agent_radius,
                              coin_radius=coin_radius,
                              signal_count=agent_vision_res,
                              fov=agent_vision_fov,
                              attenuation=agent_vision_attenuation)

    def run(self, after_init=None, on_key_press=None, on_close=None):

        # prepare first frame
        sightline = self._vision.look()
        self._draw.update(self._lvl, self._lines(sightline), self._sight_colors(sightline))

        if after_init is not None:
            threading.Thread(target=after_init, args=[self, sightline]).start()

        self._draw.run(key_handler=on_key_press, close_handler=on_close)

    def _color(self, channels):
        if channels[0] > 0:
            return self._grid_color + [channels[0]]
        elif channels[1] > 0:
            return self._coin_color + [channels[1]]
        else:
            return [0.0, 0.0, 0.0, 0.0]

    def _lines(self, signals):
        signal_lines = []
        for signal in signals:
            signal_lines.append(Line(signal.a, signal.b, self._color(signal.channels)))
        return signal_lines

    def _sight_colors(self, signals):
        colors = []
        for signal in signals:
            colors.append(self._color(signal.channels))
        return colors

    def step(self, action):
        self._ctrl.step(action)
        sightline = self._vision.look()
        self._draw.update(self._lvl, self._lines(sightline), self._sight_colors(sightline))
        return sightline
