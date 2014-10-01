import numpy as np


class ParticleFilter:


	def __init__(self, num_particles, mean, var, bounds=None):
		self.num_particles = num_particles
		self.weights = np.ones(self.num_particles)/self.num_particles
		covariance_matrix = ((var, 0),(0, var))
		self.particles = np.random.multivariate_normal(mean, covariance_matrix, self.num_particles)
		self.checkBounds(bounds)


	def resample(self):
		cum_dist = np.cumsum(self.weights)
		tmp_particles = np.zeros((self.num_particles,2))
		tmp_weights = np.zeros(self.num_particles)
		for i in range(self.num_particles):
			index = np.min(np.nonzero(np.random.rand(1) < cum_dist))
			tmp_particles[i,:] = self.particles[index,:]
			tmp_weights[i] = self.weights[index]
		self.particles = tmp_particles
		self.weights = tmp_weights


	def relocateParticles(self, mean, var, bounds=None):
		self.weights = np.ones(self.num_particles)/self.num_particles
		covariance_matrix = ((var, 0),(0, var))
		self.particles = np.random.multivariate_normal(mean, covariance_matrix, self.num_particles)
		self.checkBounds(bounds)


	def checkBounds(self, bounds):
		if (bounds != None):
			self.particles[:,0] = np.maximum(bounds[0], np.minimum(bounds[2], self.particles[:,0]))
			self.particles[:,1] = np.maximum(bounds[1], np.minimum(bounds[3], self.particles[:,1]))

	def moveGaussian(self, var=20, bounds=None):
		perturbation = var * np.random.randn(self.num_particles,2)
		self.particles += perturbation
		self.checkBounds(bounds)
	
	def updateWeights(self, update):
		self.weights = self.weights * update
		self.weights = self.weights / np.sum(self.weights)
