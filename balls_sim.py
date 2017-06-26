#!/usr/bin/env python3

import numpy as np

WORLD_LEN = 28
WORLD_SIZE = np.array((WORLD_LEN, WORLD_LEN))
N_BODIES = 3
MAX_AX_V = 1
V_STD = 0.8


def new_speeds(m1, m2, v1, v2):
    new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2


class Body(object):
    def __init__(self, pos, vel, r=4.0, m=1.0):
        self.pos = pos
        self.vel = vel
        self.r = r
        self.m = m

    def update(self, dt=1.0):
        self.pos += dt * self.vel
        # reemerging
        self.pos %= WORLD_SIZE

    def check_collision(self, other_body):
        # if centres are closer than radii
        return np.linalg.norm(self.pos - other_body.pos) < (self.r + other_body.r)


class World(object):
    def __init__(self):
        self.bodies = []
        for i in range(N_BODIES):
            self.spawn()

        # self.spawn_fake()

        space = np.linspace(0, WORLD_LEN-1, WORLD_LEN)
        self.I, self.J = np.meshgrid(space, space)

    def total_momentum(self):
        m = 0.0
        for body in self.bodies:
            m += np.linalg.norm(body.vel) * body.m

        return m

    def total_kinetic_e(self):
        ke = 0.0
        for body in self.bodies:
            ke += np.sum(body.vel**2) * body.m

        return ke/2.0

    def spawn_fake(self):
        pos1 = np.array((10.0, 10.0))
        pos2 = np.array((10.0, 20.0))
        vel1 = np.array((0, 1.001))
        vel2 = np.array((0, -1.001))

        self.bodies.append(Body(pos1, vel1))
        self.bodies.append(Body(pos2, vel2))

    def spawn(self):
        reset_required = True
        while reset_required:
            # pos = np.random.randint(0, WORLD_LEN, 2)
            pos = WORLD_LEN * np.random.rand(2)
            # vel = np.random.randint(-MAX_AX_V, MAX_AX_V+1, 2)
            vel = V_STD * np.random.randn(2)
            new_body = Body(pos, vel)

            reset_required = False
            for body in self.bodies:
                if new_body.check_collision(body):
                    reset_required = True
                    break

        self.bodies.append(new_body)

    def run(self, dt=1.0):
        for b1 in self.bodies:
            b1.pos += dt * b1.vel

            # reverse vel if wall was touched
            above_lim = (b1.pos + b1.r) > WORLD_LEN
            below_lim = (b1.pos - b1.r) < 0
            b1.vel[above_lim] = -np.abs(b1.vel[above_lim])
            b1.vel[below_lim] = np.abs(b1.vel[below_lim])

        for i, b1 in enumerate(self.bodies):
            # avoid repeating the check
            for b2 in self.bodies[i:]:
                if b1 is b2:
                    continue

                d12 = b1.pos - b2.pos
                d12_norm = np.linalg.norm(d12)
                # if collision between balls
                if d12_norm < (b1.r + b2.r):
                    m1_c = (2 * b2.m) / (b2.m + b1.m)
                    m2_c = (2 * b1.m) / (b2.m + b1.m)
                    v12 = b1.vel - b2.vel
                    v1_c = np.dot(v12, d12) * (d12/d12_norm**2)
                    v2_c = np.dot(-v12, -d12) * (-d12/d12_norm**2)

                    b1.vel -= m1_c * v1_c
                    b2.vel -= m2_c * v2_c

    def draw_centres(self):
        board = np.zeros(WORLD_SIZE)
        for body in self.bodies:
            board[int(body.pos[0]), int(body.pos[1])] = 1

        board = np.clip(board, 0, 1)
        return board.astype('uint8')

    def draw(self, obs_noise=None):

        board = np.zeros(WORLD_SIZE)
        for body in self.bodies:
            pos_x = body.pos[0]
            pos_y = body.pos[1]

            if obs_noise is not None:
                pos_x += obs_noise * np.random.randn()
                pos_y += obs_noise * np.random.randn()

            board += np.exp(-(((self.I - pos_x) ** 2 + (self.J - pos_y) ** 2) / (body.r ** 2)) ** 4)

        board = np.clip(board, 0, 1)
        return board.astype('float32')

    def give_numbers(self):
        nums = np.zeros((N_BODIES, 2, 2))
        for i, body in enumerate(self.bodies):
            nums[i, 0, :] = body.pos/WORLD_LEN
            nums[i, 1, :] = (body.vel+5)/10.0

        return nums.astype('float32')
















