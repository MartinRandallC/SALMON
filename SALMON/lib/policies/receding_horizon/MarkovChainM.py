
# coding: utf-8

# In[15]:


import numpy as np

class MarkovChain:
	def __init__( self, P):
		#P is the transition matrix of the MC
		self.dim = P.shape[0]
		# print self.dim
		self.P = P
		# self.gamma = np.zeros(self.dim)
		self.last_idx = 0
		# gamma is the actual state distribution after n_steps
		# self.gamma[self.last_idx] = 1
		# gamma_ini is the initial state distribution
		# self.gamma_ini = self.gamma
		# n_steps is the number of time slots after reset of the state distribution
		self.n_steps =0
		# the actual state is used only for montecalo simulation of the MC and is the actual state of the simulated markov process
		self.actual_state = 0
		
	def montecarlo_next_state(self):
		# for montecarlo simulation of the MC based on the actual state and the matrix P it obtains the new state
		r = np.random.choice(self.dim,1,p=self.P[self.actual_state])
		self.actual_state = r[0]
		return r[0] 
	
	def set_gamma(self, v, n_steps):
		#reset the MC to an initial state distribution and initialize n_steps 
		# self.gamma_ini = gamma_ini
		# v es el estado en el que me encuentro en la cadena
		self.last_idx = np.argmax(v)
		# print v
		# self.gamma_ini = np.zeros(self.dim)
		# self.gamma_ini[self.last_idx] = 1 
		self.n_steps = n_steps
		# self.gamma = self.gamma_ini.dot(np.linalg.matrix_power(self.P,self.n_steps)) 
		
	def step(self):
		# move the MC one step and update the state distribution gamma and also increments the steps
		self.n_steps += 1
		#print self.n_steps, self.gamma,np.linalg.matrix_power(self.P,self.n_steps)
		# self.gamma = self.gamma_ini.dot(np.linalg.matrix_power(self.P,self.n_steps))
		# return self.gamma

