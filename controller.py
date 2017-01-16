from math import cos, sin, sqrt

from enum import Enum

_enum = Enum()
ACTION_WALK_FORWARD = _enum.next()
ACTION_TURN_LEFT = _enum.next()
ACTION_TURN_RIGHT = _enum.next()

ACTIONS = (ACTION_WALK_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT)


def _walk(level, args, distance):
    agent_coord = level.agent.coord
    theta = level.agent.theta
    agent_coord[0] += distance * cos(theta)
    agent_coord[1] += distance * sin(theta)
    is_colliding = _handle_collision(level, args)
    _collect_coins(level, args)
    return is_colliding


def _handle_collision(level, args):
    grid = level.grid
    agent = level.agent
    coord = agent.coord
    x = round(coord[0])
    y = round(coord[1])
    check_cells = [[x - 1, y - 1], [x - 1, y], [x, y], [x, y - 1]]
    is_colliding = False
    for cell in check_cells:
        if _handle_bounding_collision(coord, grid, cell, args):
            is_colliding = True
    if _handle_corner_collision(coord, grid, x, y, args):
        is_colliding = True
    return is_colliding


def _handle_bounding_collision(agent_coord, grid, cell_coord, args):
    grid_shape = grid.shape

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
    ax = agent_coord[0]
    ay = agent_coord[1]
    ar = args.agent_radius

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
    agent_coord[0] = new_x
    agent_coord[1] = new_y

    return ax != new_x or ay != new_y


def _handle_corner_collision(agent_coord, grid, grid_x, grid_y, args):
    grid_shape = grid.shape

    # convex corners only exist at points surrounded by 1 'on' cell and 3 'off' cells
    # E.g.
    # tl    tr    bl    br
    # *-    -*    --    --
    # --    --    *-    -*

    # grid dimensions
    h = grid_shape[0]
    w = grid_shape[1]

    # grid is flipped on y-axis
    bot = h - grid_y
    top = bot - 1
    rgt = grid_x
    lft = rgt - 1

    # be careful not to look outside the grid dimensions
    tl = grid[top, lft] if 0 <= top < h and 0 <= lft < w else 0
    tr = grid[top, rgt] if 0 <= top < h and 0 <= rgt < w else 0
    bl = grid[bot, lft] if 0 <= bot < h and 0 <= lft < w else 0
    br = grid[bot, rgt] if 0 <= bot < h and 0 <= rgt < w else 0

    # exit early if this isn't a corner
    if tl + tr + bl + br != 1:
        return False

    # agent coordinates and radius, radius ^ 2
    ax = agent_coord[0]
    ay = agent_coord[1]
    ar = args.agent_radius
    ar2 = ar * ar

    # adjusted agent coordinates
    new_x = ax;
    new_y = ay;

    # calculate distance to agent
    vx, vy = ax - grid_x, ay - grid_y
    dist2 = vx * vx + vy * vy

    # colliding?
    if dist2 < ar2:
        # avoid costly sqrt until we're sure we have to
        dist = sqrt(dist2)

        # rescale vector with magnitude equal to agent's radius
        vxr = vx * ar / dist
        vyr = vy * ar / dist

        # apply adjustment to both axes
        new_x = grid_x + vxr
        new_y = grid_y + vyr

    # apply new agent coordinates
    agent_coord[0] = new_x
    agent_coord[1] = new_y

    return ax != new_x or ay != new_y


def _collect_coins(level, args):
    coins = level.coins
    agent_coord = level.agent.coord
    agent_x = agent_coord[0]
    agent_y = agent_coord[1]

    # threshold distance for coin collection (squared)
    threshold = args.agent_radius + args.coin_radius
    threshold2 = threshold * threshold

    i = 0
    while i < len(coins):

        # coin coordinates
        coin = coins[i]
        cx, cy = coin[0], coin[1]

        # agent-to-coin vector
        vx, vy = cx - agent_x, cy - agent_y

        # close enough to collect?
        dist2 = vx * vx + vy * vy
        if dist2 < threshold2:
            # "collect" the coin by deleting it
            del coins[i]

        i += 1


class Controller:
    def __init__(self, args, level):
        self._args = args
        self._level = level

    def step(self, action):
        is_colliding = False
        if action == ACTION_WALK_FORWARD:
            is_colliding = _walk(self._level, self._args, self._args.agent_stride)
        elif action == ACTION_TURN_LEFT:
            self._level.agent.theta += self._args.agent_turn
            is_colliding = _walk(self._level, self._args, self._args.agent_stride_on_turn)
        elif action == ACTION_TURN_RIGHT:
            self._level.agent.theta -= self._args.agent_turn
            is_colliding = _walk(self._level, self._args, self._args.agent_stride_on_turn)
        return is_colliding
