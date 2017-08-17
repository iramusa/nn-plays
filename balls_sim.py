#!/usr/bin/env python3

import numpy as np

WORLD_LEN = 28
WORLD_SIZE = np.array((WORLD_LEN, WORLD_LEN))
V_STD = 0.8

default_config = {
    'n_bodies': 1,
    'radius_mode': 'uniform',
    'radius': 3.5,
    'mass_mode': 'uniform',
    'mass': 1.0,
    'wall_action': 'pass',
    'ball_action': 'pass',
    'measurement_noise': 0.0,
    'dynamics_noise': 0.001,
}


class Body(object):
    def __init__(self, pos, vel, r=4.0, m=1.0):
        self.pos = pos
        self.measured_pos = np.copy(pos)
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
    def __init__(self, **kwargs):
        self.n_bodies = kwargs['n_bodies']
        self.radius_mode = kwargs['radius_mode']
        self.mass_mode = kwargs['mass_mode']
        self.wall_action = kwargs['wall_action']
        self.ball_action = kwargs['ball_action']

        self.measurement_noise = kwargs['measurement_noise']
        self.dynamics_noise = kwargs['dynamics_noise']

        radius = kwargs['radius']
        if self.radius_mode == 'uniform':
            self.radii = [radius for _ in range(self.n_bodies)]
        else:
            raise ValueError('Bad radius_mode', self.radius_mode)

        mass = kwargs.get('mass', 1.0)
        if self.radius_mode == 'uniform':
            self.masses = [mass for _ in range(self.n_bodies)]
        elif self.radius_mode == 'radius_tied':
            # mass proportional to r^2
            self.masses = [r**2 for r in self.radii]
        else:
            raise ValueError('Bad mass_mode', self.mass_mode)

        # spawn bodies
        self.bodies = []
        for i in range(self.n_bodies):
            self.spawn(radius=self.radii[i], mass=self.masses[i])

        # self.spawn_fake()

        space = np.linspace(0.5, WORLD_LEN-0.5, WORLD_LEN)
        self.I, self.J = np.meshgrid(space, space)

    def total_momentum(self):
        """Total momentum should be constant if balls pass through walls

        :return: m -- total momentum in system
        """
        m = 0.0
        for body in self.bodies:
            m += np.sum(body.vel * body.m)

        return m

    def total_kinetic_e(self):
        """Total kinetic energy should be constant all the time (elastic collisions).

        :return:
        """
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

    def spawn(self, radius, mass):
        reset_required = True
        while reset_required:
            # pos = np.random.randint(0, WORLD_LEN, 2)
            pos = WORLD_LEN * np.random.rand(2)
            vel = V_STD * np.random.randn(2)
            new_body = Body(pos, vel, r=radius, m=mass)

            reset_required = False

            if np.any((new_body.pos - new_body.r) < 0) or np.any((new_body.pos + new_body.r) > WORLD_LEN):
                reset_required = True
                continue

            for body in self.bodies:
                if new_body.check_collision(body):
                    reset_required = True
                    break

        self.bodies.append(new_body)

    def run(self, dt=1.0):
        # state and measurement update
        for b1 in self.bodies:
            dv = self.dynamics_noise * np.random.randn(2)
            b1.pos += dt * b1.vel + (dv * dt**2 / 2)
            b1.vel += dv*dt
            b1.measured_pos = np.copy(b1.pos)

            if self.measurement_noise == 0.0:
                pass
            else:
                # add noise
                b1.measured_pos += self.measurement_noise * np.random.randn(2)

        # wall action
        for b1 in self.bodies:
            if self.wall_action == 'pass':
                # reappear target on other side
                b1.pos %= WORLD_LEN
            elif self.wall_action == 'bounce':
                # reverse vel if wall was touched
                above_lim = (b1.pos + b1.r) > WORLD_LEN
                below_lim = (b1.pos - b1.r) < 0
                b1.vel[above_lim] = -np.abs(b1.vel[above_lim])
                b1.vel[below_lim] = np.abs(b1.vel[below_lim])
            else:
                raise ValueError('Bad wall_action', self.wall_action)

        # ball action
        if self.ball_action == 'pass':
            pass
        elif self.ball_action == 'bounce':
            # bounce balls
            for i, b1 in enumerate(self.bodies):
                # avoid repeating the check
                for b2 in self.bodies[i:]:
                    if b1 is b2:
                        continue

                    d12 = b1.pos - b2.pos
                    d12_norm = np.linalg.norm(d12)
                    if d12_norm < (b1.r + b2.r):
                        # if collision between balls
                        m1_c = (2 * b2.m) / (b2.m + b1.m)
                        m2_c = (2 * b1.m) / (b2.m + b1.m)
                        v12 = b1.vel - b2.vel
                        v1_c = np.dot(v12, d12) * (d12 / d12_norm ** 2)
                        v2_c = np.dot(-v12, -d12) * (-d12 / d12_norm ** 2)

                        b1.vel -= m1_c * v1_c
                        b2.vel -= m2_c * v2_c
        else:
            raise ValueError('Bad ball_action', self.ball_action)

    def draw_centres(self):
        board = np.zeros(WORLD_SIZE)
        for body in self.bodies:
            board[int(body.pos[0]), int(body.pos[1])] = 1

        board = np.clip(board, 0, 1)
        return board.astype('uint8')

    def draw(self, obs_noise=None):

        board = np.zeros(WORLD_SIZE)
        for body in self.bodies:
            pos_x = body.measured_pos[0]
            pos_y = body.measured_pos[1]

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
















