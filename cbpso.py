#!/usr/bin/env python3
#
# By Sebastian Raaphorst, 2019.
#
# Complimentary binary particle swarm optimization algorithm, as per:
# Li-Yeh Chuang, Chih-Jen Hsiao, Cheng-Hong Yang, An Improved Binary Particle Swarm Optimization with Complementary
# Distribution Strategy for Feature Selection, 2009 International Conference on Machine Learning and Computing.
# https://pdfs.semanticscholar.org/7b5e/7a8b7f86366e446ecd37121e5cd20199d329.pdf

import numpy as np
from math import exp


class CBPSO:
    def __init__(self,
                 dimensions,
                 fitness,
                 stop_func=lambda _: False,
                 num_particles=50,
                 rounds=100,
                 max_rounds_to_complement=3,
                 w=1,
                 c1=1,
                 c2=1,
                 vmax=1,
                 end_round=None):
        self._dimensions = dimensions
        self._fitness = fitness
        self._stop_func = stop_func
        self._num_particles = num_particles
        self._particles = None
        self._rounds = rounds
        self._max_rounds_to_complement = max_rounds_to_complement
        self._w = w
        self._c1 = c1
        self._c2 = c2
        self._vmax = vmax
        self._end_round = end_round

    @staticmethod
    def _S(x):
        return 1.0 / (1.0 + exp(-x))

    def run(self):
        # Shorthand to avoid self._ references:
        p, d = self._num_particles, self._dimensions

        # Initialize all the particle positions and record as the best positions.
        self._particles = np.random.randint(0, 2, (p, d))
        best_particles = np.zeros((p, d))
        best_fitnesses = np.zeros(p)

        # Structures to keep track of the global best particle.
        best_particle = None
        best_particle_fitness = None

        # Randomize the velocity dimensions to be between [-vmax, vmax].
        velocities = (np.random.random((p, d)) * 2 - 1) * self._vmax

        # Keep track of the number of rounds where the best_particle has not changed.
        num_rounds = 0

        for t in range(self._rounds):
            if best_particle is not None:
                print("Round {}: {}".format(t, best_particle_fitness))
            # Calculate the fitness of the particles and update the best fitnesses and best_particles if we
            # have improved.
            fitnesses = np.apply_along_axis(self._fitness, 1, self._particles)
            fitness_comparison = fitnesses > best_fitnesses
            best_particles = np.where(fitness_comparison.reshape(p, 1), self._particles, best_particles)
            best_fitnesses = np.where(fitness_comparison, fitnesses, best_fitnesses)

            # Keep track of the best particle.
            best_particle_candidate = np.argmax(best_fitnesses)
            best_particle_candidate_fitness = best_fitnesses[best_particle_candidate]
            if best_particle is None or best_particle_fitness < best_particle_candidate_fitness:
                best_particle, best_particle_fitness = best_particle_candidate, best_particle_candidate_fitness
                num_rounds = 0
            else:
                num_rounds += 1

            # If we have not changed, take the complement of a random half of the vectors.
            # There is probably a numpy guru way to do this without iterating, but I'm not sure what it is.
            # With fancy indexing, we can do this in one step.
            if num_rounds >= self._max_rounds_to_complement:
                rows = np.random.choice(p, p // 2)
                self._particles[[rows], :] = 1 - self._particles[[rows], :]
                num_rounds = 0

            # Update the particle velocities.
            velocities = self._w * velocities +\
                         self._c1 * np.random.random((p, d)) * (np.tile(best_fitnesses, d).reshape((p, d))
                                                                - np.tile(fitnesses, d).reshape(p, d)) +\
                         self._c2 * np.random.random((p, d)) * (np.tile(np.tile(best_particle, p), d).reshape(p, d)
                                                                - np.tile(fitnesses, d).reshape(p, d))

            # Clamp them to being in [-vmax, vmax].
            velocities = np.where(np.where(velocities < -self._vmax, -self._vmax, velocities) > self._vmax,
                                  self._vmax, velocities)

            # Now calculate the new positions.
            self._particles = np.where(np.random.random((p, d)) < np.vectorize(CBPSO._S)(velocities), 1, 0)

            # Check the termination condition.
            if best_particle is not None and self._stop_func(self._particles[best_particle]):
                return True, self._particles[best_particle], best_particle_fitness

        # Failure.
        return False, self._particles[best_particle], best_particle_fitness

