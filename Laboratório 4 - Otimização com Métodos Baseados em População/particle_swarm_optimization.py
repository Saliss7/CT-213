import numpy as np
import random
from math import inf


class Particle:
    """
    Represents a particle of the Particle Swarm Optimization algorithm.
    """
    def __init__(self, lower_bound, upper_bound):
        """
        Creates a particle of the Particle Swarm Optimization algorithm.

        :param lower_bound: lower bound of the particle position.
        :type lower_bound: numpy array.
        :param upper_bound: upper bound of the particle position.
        :type upper_bound: numpy array.
        """
        self.best_position = np.zeros(np.size(lower_bound))
        self.best_value = -inf
        self.x = np.random.uniform(lower_bound, upper_bound)
        self.v = np.random.uniform(-(upper_bound - lower_bound), upper_bound - lower_bound)


class ParticleSwarmOptimization:
    """
    Represents the Particle Swarm Optimization algorithm.
    Hyperparameters:
        inertia_weight: inertia weight.
        cognitive_parameter: cognitive parameter.
        social_parameter: social parameter.

    :param hyperparams: hyperparameters used by Particle Swarm Optimization.
    :type hyperparams: Params.
    :param lower_bound: lower bound of particle position.
    :type lower_bound: numpy array.
    :param upper_bound: upper bound of particle position.
    :type upper_bound: numpy array.
    """
    def __init__(self, hyperparams, lower_bound, upper_bound):
        self.particles = [
            Particle(lower_bound, upper_bound)
            for _ in range(hyperparams.num_particles)
        ]
        self.current_particle_index = 0
        self.best_global_position = np.zeros(np.size(lower_bound))
        self.best_global_value = -inf
        self.num_particles = hyperparams.num_particles
        self.inertia_weight = hyperparams.inertia_weight
        self.cognitive_parameter = hyperparams.cognitive_parameter
        self.social_parameter = hyperparams.social_parameter
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_best_position(self):
        """
        Obtains the best position so far found by the algorithm.

        :return: the best position.
        :rtype: numpy array.
        """
        return self.best_global_position

    def get_best_value(self):
        """
        Obtains the value of the best position so far found by the algorithm.

        :return: value of the best position.
        :rtype: float.
        """
        return self.best_global_value

    def get_position_to_evaluate(self):
        """
        Obtains a new position to evaluate.

        :return: position to evaluate.
        :rtype: numpy array.
        """
        return self.particles[self.current_particle_index].x

    def advance_generation(self):
        """
        Advances the generation of particles. Auxiliary method to be used by notify_evaluation().
        """
        for i in range(self.num_particles):
            rp = np.random.uniform(0.0, 1.0)
            rg = np.random.uniform(0.0, 1.0)

            v_new = (self.inertia_weight * self.particles[i].v +
                     self.cognitive_parameter * rp * (self.particles[i].best_position - self.particles[i].x) +
                     self.social_parameter * rg * (self.best_global_position - self.particles[i].x))

            self.particles[i].v = np.clip(v_new, -(self.upper_bound - self.lower_bound), self.upper_bound - self.lower_bound)

            self.particles[i].x = np.clip(self.particles[i].x + self.particles[i].v, self.lower_bound, self.upper_bound)

    def notify_evaluation(self, value):
        """
        Notifies the algorithm that a particle position evaluation was completed.

        :param value: quality of the particle position.
        :type value: float.
        """
        if self.particles[self.current_particle_index].best_value < value:
            self.particles[self.current_particle_index].best_position = self.particles[self.current_particle_index].x
            self.particles[self.current_particle_index].best_value = value
        if self.best_global_value < value:
            self.best_global_position = self.particles[self.current_particle_index].x
            self.best_global_value = value

        self.current_particle_index = (self.current_particle_index + 1) % self.num_particles

        if self.current_particle_index == 0:
            self.advance_generation()
