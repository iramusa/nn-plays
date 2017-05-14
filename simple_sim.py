#!/usr/bin/env python3

import numpy as np
from scipy.signal import convolve2d

WORLD_LEN = 28
WORLD_SIZE = np.array((WORLD_LEN, WORLD_LEN))
# N_BODIES = 4
N_BODIES = 1
MAX_AX_V = 1

FACES = {
    '[]': np.array([
        [1, 1, 1, 1, 1,],
        [1, 1, 1, 1, 1,],
        [1, 1, 1, 1, 1,],
        [1, 1, 1, 1, 1,],
        [1, 1, 1, 1, 1,],
    ]),
    # 'T': np.array([
    #     [1, 1, 1],
    #     [0, 1, 0],
    #     [0, 1, 0],
    # ]),
    # '/': np.array([
    #     [0, 0, 1],
    #     [0, 1, 0],
    #     [1, 0, 0],
    # ]),
    # '-': np.array([
    #     [0, 0, 0],
    #     [1, 1, 1],
    #     [0, 0, 0],
    # ]),
}
FACE_LIST = FACES.keys()


class Body(object):
    def __init__(self, pos, vel, face_type):
        self.pos = pos
        self.vel = vel
        self.face_type = face_type

    def update(self):
        self.pos += self.vel
        # reemerging
        self.pos %= WORLD_SIZE


class World(object):
    def __init__(self):
        self.bodies = []
        for i in range(N_BODIES):
            self.spawn()

    def spawn(self):
        pos = np.random.randint(0, WORLD_LEN, 2)
        vel = np.random.randint(-MAX_AX_V, MAX_AX_V+1, 2)
        face_type = list(FACE_LIST)[np.random.randint(len(FACE_LIST))]
        new_body = Body(pos, vel, face_type)
        self.bodies.append(new_body)

    def run(self):
        for body in self.bodies:
            body.update()

    def draw(self):
        board = np.zeros(WORLD_SIZE)
        for face in FACE_LIST:
            tmp_board = np.zeros(WORLD_SIZE)
            for body in self.bodies:
                if body.face_type == face:
                    tmp_board[body.pos[0], body.pos[1]] = 1

            tmp_board = convolve2d(tmp_board, FACES[face], mode='same')
            board += tmp_board

        board = np.clip(board, 0, 1)
        return board.astype('uint8')
















