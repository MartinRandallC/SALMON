# coding: utf-8

import numpy as np
import random
import itertools
#import MarkovChainM as mc
#from NetworkEnvironmentM import *
from . import MarkovChainM as mc
from .NetworkEnvironmentM import *

# for further recursion!
sys.setrecursionlimit(10000)

# The system is a set of routes. Each route has a Markov chain with it qos model.
# The state of the system is the set of the vector of states distribution of the markov chain of each route
# The actions on each route are: measure the sate of the route (1) or not.
# The receding algoritm will allow us to know where to measure or not, and which route(s)
class RecedingHorizon(object):
	def __init__(self, matrixPtot, qos, costs, max_slots, discount_factor, depth, one_route=False, pruning=0, nomemory=True):
		# matrixPtot is a list of numpy arrays each one is the probability transition Matrix with the dimensions of the route markov model.
		# qos is a list of vectors, one for each route, and each vector has the delays in each state.
		# costs is a list of the cost of measure in each route.
		# max_slots is the maximum number of time slots before mandatory measure/reset in each route
		# discount_factor is the discount factor for the receding horizon algoritm
		# depth is the horizon of the algoritm
		# one_route is to measure at most one route per time step if set on true, and the restriction is lifted if set on false
		# pruning is to discard next states with probability lesser than pruning (if set on 0 we don't do pruning)
		# nomemory is to reset the Q-value dictionnary if set on True, and to keep it if set on False
		# Q is the dictionnary with Qvalues(state)=[Qsa1, Qsa2, Qsa3 ...] for state s and action a1, a2, etc.
		# ne is the Network environment, where we recreate the MDP.
		self.Psim = matrixPtot
		self.qos = qos
		self.max_slots = max_slots
		self.ro = discount_factor
		self.H = depth
		self.one_route = one_route
		self.pruning = pruning
		self.nomemory = nomemory
		self.costs_original = costs
		self.ne = NetworkEnvironmentIntState(self.Psim, self.qos, costs, self.max_slots, self.one_route)
		self.costs=self.ne.costs
		# the actions are to measure or not in each route
		self.n_actions = self.ne.n_actions#1+len(qos)#n_routes
		self.action={}
		if self.one_route:
			self.action[0]=self.ne.convert_int_to_action(0)
			for i in range(1, self.n_actions):
				self.action[i]=self.ne.convert_int_to_action(2**(i-1))
			# print(self.action)
		else:
			for i in range(self.n_actions):
				self.action[i]=self.ne.convert_int_to_action(i)
		# print(self.action)
		# print(self.n_actions)
		self.dic=self.DicMat()
		# print (self.dic)
		self.Q={}
		# ini_state=self.ne.current_state()
		# print( ini_state)
		if self.nomemory!=True:
			self.H+=1
			self.RH(self.ne.current_state())
			self.H=depth
		# print (self.Q)
		# auxiliar
		self.contador=0
		# self.Q2={}
		# self.n_states = self.states_number()
		# states = [] # the set of all possible states
		# for r in range(self.ne.n_routes):
		# 	st = []
		# 	lag = []
		# 	for s in range(self.ne.mcs[r].dim):
		# 		st.append(s)
		# 	for l in range(self.max_slots[r]-self.H):
		# 		lag.append(l)
		# 	states.append(st)
		# 	states.append(lag)
		# self.states = tuple(itertools.product(*states))
		# print self.states

	#########################################
	# Problema de optimización
	#########################################
	# le paso los costos ya como me interesan y no lo hago cada vez
	def ProblemaLineal(self):
		# hallo los D para todos los a posibles
		# ahora paso un vector de accion directamente
		# que es a priori todo 0 y vamos rotando un 1 que es la ruta a medir
		D=[]
		Mat=[]
		# Mat2=[]
		if self.one_route:
			D.append(self.calculo_D(0))
			Mat.append(D[0])
# 			Mat.append(-self.ne.R(self.ne.current_state(),0))
			for i in range(self.ne.n_routes):
				D.append(self.calculo_D(i+1))
				Mat.append(D[i+1]+self.ne.costs[i])
				# print(D[i+1]+self.ne.costs[i])
				# print(self.ne.R(self.ne.current_state(),i+1))
# 				Mat.append(-self.ne.R(self.ne.current_state(),i+1))
		else:
			for i in range(self.ne.n_actions):
				Mat.append(-self.ne.R(self.ne.current_state(),i))
# 				D.append(self.calculo_D(i))
# 			Mat = D+self.ne.costs
		# print ('valor final de Matriz R: ', Mat)
		# print ('valor final de Matriz D: ', Mat2)
		return Mat

	# obtengo todos los valores medidos y estimados:
	def obtener_estados(self, a):
		# quiero devolver en k un diccionario que este ordenado con {l*:P*, l**:P**...}
		k={}
		# print ('accion pedida: ', a)
		# a_bin = self.ne.convert_int_to_action(a)
		# print(a_bin)
		a_bin=self.action[a]
		# print 'accion posible: ', a_bin
		maxmins=[]
		for i in range(self.ne.n_routes):
			maxmins.append(max(self.ne.qos[i]))
		maxmin = min(maxmins)
		for i in range(self.ne.n_routes):
			# x=np.zeros(len(self.ne.qos[i]))
			# x[self.ne.mcs[i].last_idx]=1
			# b=x.dot(self.dic[i, self.ne.mcs[i].n_steps])
			b=self.dic[i, self.ne.mcs[i].n_steps][self.ne.mcs[i].last_idx]
			# print('valor de b: ', b)
			# si toca medir en esa ruta:
			if a_bin[i]==1:# or self.ne.mcs[i].n_steps==(self.max_slots[i]-1):
				for j in range(len(self.ne.qos[i])):
					k.update({'P'+str(i)+'K'+str(j):[self.ne.qos[i][j],b[j]]})
			else:
				L=0
				for j in range(len(self.ne.qos[i])):
					L+=b[j]*self.ne.qos[i][j]
				k.update({'SM-P'+str(i):[L, 1]})
		return k, maxmin

	def calculo_D(self, a):
		# Obtengo un diccionario con los valores {ruta: [l*, P*]}
		k, maxmin2 = self.obtener_estados(a)
		# print (str(self.ne.current_state()) + ' ' + str(a))
		# print ('estados [l*, P*] ', k)#, maxmin2
		# calculo el D
		# D=[]
		D=0
		ordenada=sorted(k.values()) #ordenada[0][0] el l* y en [0][1] el P*
		# print (ordenada)
		maxmin=0
		bandera2=False
		# caso en que maxmin es un valor medio:
		l=0
		while bandera2==False and l<len(ordenada):
			if ordenada[l][1]==1:
				maxmin=ordenada[l][0]
				bandera2=True
			l+=1
		# maxmin es un valor medido:
		if maxmin>maxmin2 or maxmin==0:
			maxmin=maxmin2
		#print bandera2
		#print '/// El minimo de los maximos de los retardos entre medidos y medios es: ', maxmin
		#print ordenada
		bandera=False
		i=0
		while bandera==False and i<len(ordenada):
			if i==0:
				# D.append(ordenada[i][0]*ordenada[i][1])
				D+=ordenada[i][0]*ordenada[i][1]
			elif ordenada[i][0]==maxmin:
				aux=1
				for j in range(i):
					aux*=(1-ordenada[j][1])
				# D.append(ordenada[i][0]*aux)
				D+=ordenada[i][0]*aux
				bandera=True
			else:
				aux=1
				for j in range(i):
					aux*=(1-ordenada[j][1])
				# D.append(ordenada[i][0]*ordenada[i][1]*aux)
				D+=ordenada[i][0]*ordenada[i][1]*aux
			# print(D)
			i+=1
		# D=sum(D)
		# print ('Valor de D ', D ,' para la combinatoria ', a)
		# print('Valor de R ', self.ne.R(self.ne.current_state(), a))
		# print('Valor de T ', self.ne.T(self.ne.current_state(), a, self.dic))
		# print('Transiciones en RH: ', self.possible_states())
		return D

	################
	# Funcion para setear estado
	################
	def setear_estado(self, estado_deseado):
		# print estado_deseado
		# ahora estado_deseado es un array de largo cantidad de rutas
		# en el que tengo el estado que me interesa poner en 1
		for i in range(len(self.ne.mcs)):
			v = np.zeros(self.ne.mcs[i].dim)
			v[estado_deseado[2*i]] = 1
			# print gamma_ini
			self.ne.mcs[i].set_gamma(v, estado_deseado[2*i+1])
		self.DicUpdate()
		# print self.current_state()

	##################################################
	# Creo un diccionario con las matmult de Psim**lag
	##################################################
	def DicMat(self):
		dic={}
		# este diccionario se pasa luego a ne.T que lo usa en vez de hacer los calculos de potencias
		# aca tengo diccionario[ruta, lag]=Pruta**(lag+1)
		for j in range(self.ne.n_routes):
			Tau0=self.ne.mcs[j].n_steps
			for i in range(self.H+1):
				dic[j, i]=np.linalg.matrix_power(self.ne.mcs[j].P, i+1)
			# for i in range(max(int(Tau0), self.H), self.H+int(Tau0)):
			# 	dic[j, i]=np.linalg.matrix_power(self.ne.mcs[j].P, int(Tau0+i+1))
		return dic

	########################################################################
	# Creo un diccionario con las matmult de Psim**lag ahora centrado en Tau
	########################################################################
	def DicUpdate(self):
		# este diccionario se pasa luego a ne.T que lo usa en vez de hacer los calculos de potencias
		for j in range(self.ne.n_routes):
			Tau0=self.ne.mcs[j].n_steps
			for i in range(max(int(Tau0), self.H), self.H+int(Tau0)+1):
				self.dic[j, i]=np.linalg.matrix_power(self.ne.mcs[j].P, int(i+1))
		return self.dic


	#################
	# Funcion que devuelve un diccionario con todos los posibles estados y su probabilidad de transicion a partir de un estado
	#################
	def possible_states(self):
		# ne is the actual state
		# dic is the P**Tau dictionnary for all possible states
		# the function returns future_states=[accion: [ne1, Pne1], [ne2, Pne2], etc] possible states and probabilities for each action
		# print ("Estado actual: " + str(ne.current_state()))
		lista_ordenada=[]
		if self.one_route:
			lista_ordenada.append(self.ne.T(self.ne.current_state(), 0, self.dic))
			for i in range(self.ne.n_routes):
				lista_ordenada.append(self.ne.T(self.ne.current_state(), 2**i, self.dic))
		else:
			for i in range(self.ne.n_actions):
				lista_ordenada.append(self.ne.T(self.ne.current_state(), i, self.dic))
		# print(lista_ordenada)
		return lista_ordenada

	#################
	# Funcion que devuelve un diccionario con todos los posibles estados y su probabilidad de transicion a partir de un estado, K pasos para adelante
	#################
	def recedingQ(self, H):#, Q, Q2):
		# returns V(s)*, a*
		estado_original=self.ne.current_state()
		# print(estado_original)
		# # if H>1:
		for h in range(H, self.H):
			if (estado_original, h) in self.Q:
				return self.Q[(estado_original, h)]
		if H==1:
			D=self.ProblemaLineal()
			self.Q[(estado_original, 1)]=[np.min(D), np.argmin(D)]
			return self.Q[(estado_original, 1)]
		elif H==0:
			self.contador+=1
			return [0, 0]
		estados_ordenados=self.possible_states()
		D=self.ProblemaLineal()
		# print(self.Q2[(estado_original, 1)])
		# para cada accion (indice)
		suma=np.zeros(self.ne.n_actions)
		for i in range(len(estados_ordenados)):
			for j in estados_ordenados[i]:
				if j[1]<min(self.ro**self.H, self.pruning):
					continue
				# print j[0]
				self.setear_estado(j[0])
				self.Q[(j[0], H-1)]=self.recedingQ(H-1)
				# print(self.Q[(j[0], H-1)])
				suma[i]+=j[1]*self.Q[(j[0], H-1)][0]
		#print estado_original
		resultado=D+self.ro*suma
		# accion=np.argmin(self.Q[(estado_original, H)])
		# self.Q[(estado_original, H)]=[np.min(self.Q[(estado_original, H)]), accion]
		self.Q[(estado_original, H)]=[np.min(resultado), np.argmin(resultado)]
		# if H==self.H:
		# 	print(len(self.Q))
		# 	print(self.contador)
		return self.Q[(estado_original, H)]

	def RH(self, new_state):#, Q, Q2):
		if self.nomemory==True:
			self.Q={}
		# new_state: we calculate the receding horizon for self.H steps from the state new_state
		self.setear_estado(new_state)
		# self.DicUpdate()
		[Q, a]=self.recedingQ(self.H)
		# sol=self.ne.convert_int_to_action(np.argmin(Q))
		sol=self.action[a]
		return sol#a, Q
