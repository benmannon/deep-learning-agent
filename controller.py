from math import cos, sin, sqrt

import numpy as np

walk_forward = 0
turn_left = 1
turn_right = 2

actions = [walk_forward, turn_left, turn_right]


class Controller:
    def __init__(self, level, agent_stride, agent_turn, agent_radius=0.45, coin_radius=0.25):
        self._level = level
        self._grid_shape = np.array(level.grid).shape
        self._stride = agent_stride
        self._turn = agent_turn
        self._agent_radius = agent_radius
        self._coin_radius = coin_radius

    def step(self, action):
        is_colliding = False
        if action == walk_forward:
            is_colliding = self._walk(self._level.agent.theta, self._stride)
        elif action == turn_left:
            self._level.agent.theta += self._turn
        elif action == turn_right:
            self._level.agent.theta -= self._turn
        return is_colliding

    def _walk(self, theta, distance):
        agent = self._level.agent
        x, y = agent.coord[0], agent.coord[1]
        agent.coord = [x + distance * cos(theta), y + distance * sin(theta)]
        is_colliding = self._handle_collision()
        self._collect_coins()
        return is_colliding

    def _handle_collision(self):
        level = self._level
        grid = level.grid
        agent = level.agent
        coord = agent.coord
        x = round(coord[0])
        y = round(coord[1])
        check_cells = [[x - 1, y - 1], [x - 1, y], [x, y], [x, y - 1]]
        is_colliding = False
        for cell in check_cells:
            if self._handle_bounding_collision(cell):
                is_colliding = True
        for cell in check_cells:
            if self._handle_corner_collision(cell):
                is_colliding = True
        return is_colliding

    def _handle_bounding_collision(self, cell_coord):

        grid_shape = self._grid_shape
        grid = self._level.grid

        # cell coordinates
        cx = cell_coord[0]
        cy = cell_coord[1]

        # is the cell out of bounds? exit early
        if cx < 0 or cy < 0 or cx >= grid_shape[1] or cy >= grid_shape[0]:
            return

        # is the cell open? exit early
        # (y-axis on grid is flipped)
        cy_flip = grid_shape[0] - cy - 1
        cell = grid[cy_flip][cx]
        if cell == 0:
            return

        # the cell's bounding box vertices
        cx0, cx1 = cx, cx + 1
        cy0, cy1 = cy, cy + 1

        # cell center point
        ccx = (cx0 + cx1) / 2
        ccy = (cy0 + cy1) / 2

        # agent coordinates and radius
        agent = self._level.agent
        ax = agent.coord[0]
        ay = agent.coord[1]
        ar = self._agent_radius

        # adjusted agent coordinates
        new_x = ax;
        new_y = ay;

        # calculate adjustment on the y-axis
        if cx0 <= ax <= cx1:
            if ay > ccy and ay - cy1 < ar:
                new_y = cy1 + ar
            elif ay < ccy and cy0 - ay < ar:
                new_y = cy0 - ar

        # calculate adjustment on the x-axis
        if cy0 <= ay <= cy1:
            if ax > ccx and ax - cx1 < ar:
                new_x = cx1 + ar
            elif ax < ccx and cx0 - ax < ar:
                new_x = cx0 - ar

        # apply new agent coordinates
        agent.coord[0] = new_x
        agent.coord[1] = new_y

        return ax != new_x or ay != new_y

    def _handle_corner_collision(self, cell_coord):

        grid_shape = self._grid_shape
        grid = self._level.grid

        # cell coordinates
        cx = cell_coord[0]
        cy = cell_coord[1]

        # is the cell out of bounds? exit early
        if cx < 0 or cy < 0 or cx >= grid_shape[1] or cy >= grid_shape[0]:
            return

        # is the cell open? exit early
        # (y-axis on grid is flipped)
        cy_flip = grid_shape[0] - cy - 1
        cell = grid[cy_flip][cx]
        if cell == 0:
            return

        # the cell's bounding box vertices
        cx0, cx1 = cx, cx + 1
        cy0, cy1 = cy, cy + 1

        # cell center point
        ccx = (cx0 + cx1) / 2
        ccy = (cy0 + cy1) / 2

        # agent coordinates and radius
        agent = self._level.agent
        ax = agent.coord[0]
        ay = agent.coord[1]
        ar = self._agent_radius

        # adjusted agent coordinates
        new_x = ax;
        new_y = ay;

        # find the nearest corner
        ncx = cx0 if ax <= ccx else cx1
        ncy = cy0 if ay <= ccy else cy1

        # calculate distance to agent
        vx, vy = ax - ncx, ay - ncy
        dist = sqrt(vx * vx + vy * vy)

        # colliding?
        if dist < ar:
            # rescale vector with magnitude equal to agent's radius
            vxr = vx * ar / dist
            vyr = vy * ar / dist

            # apply adjustment to both axes
            new_x = ncx + vxr
            new_y = ncy + vyr

        # apply new agent coordinates
        agent.coord[0] = new_x
        agent.coord[1] = new_y

        return ax != new_x or ay != new_y

    def _collect_coins(self):

        # agent coordinate
        agent = self._level.agent
        ax = agent.coord[0]
        ay = agent.coord[1]

        # threshold distance for coin collection
        threshold = self._agent_radius + self._coin_radius

        coins = self._level.coins
        i = 0
        while i < len(coins):

            # coin coordinates
            coin = coins[i]
            cx, cy = coin[0], coin[1]

            # agent-to-coin vector
            vx, vy = cx - ax, cy - ay

            # close enough to collect?
            dist = sqrt(vx * vx + vy * vy)
            if dist < threshold:
                # "collect" the coin by deleting it
                del coins[i]

            i += 1
