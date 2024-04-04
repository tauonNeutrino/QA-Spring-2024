#%% Python 3.12.1

import json
import dimod
import math
from collections import defaultdict
from dwave.system import LeapHybridSampler, DWaveSampler, EmbeddingComposite
# import dwavebinarycsp
import dwave.inspector
import numpy as np
import matplotlib.pyplot as plt

#%%

def get_rand_from(v):
	elem = v[np.random.randint(len(v))]
	v.remove(elem)
	return elem

# maybe base this on https://arxiv.org/pdf/1405.6569.pdf in the future
def generate_clusters():

	z_range = 1 # 0 to 1
	theta_range = 1 # 0 to 1 (we're acting like we squished the ranges)
	num_tracks = 12

	stdev = 0.01

	p = np.random.rand(num_tracks)
	p = .1/(p**2 + 0.01) # some p distribution I made up
	p /= p.max()
	n = sum(p > 0.3)
	print(n)
	if n > 5 or n < 2:
	# if n != 5: # just for now
		return generate_clusters()
	
	p = np.sort(p)[::-1].tolist()

	hard_ps = p[:n]
	soft_ps = p[n:]

	vertex_zs = np.random.rand(n) * z_range
	vertex_thetas = np.random.rand(n) * theta_range

	points_per_cluster = (num_tracks // n) - 1 # -1 because we're going to add the vertex itself

	all_zs = []
	all_thetas = []
	all_ps = []
	for i in range(n):
		z = vertex_zs[i]
		theta = vertex_thetas[i]
		for j in range(points_per_cluster):
			all_zs.append(z + np.random.normal(scale=stdev))
			all_thetas.append(theta + np.random.normal(scale=stdev))
			all_ps.append(get_rand_from(soft_ps))
		all_zs.append(z)
		all_thetas.append(theta)
		all_ps.append(get_rand_from(hard_ps))
	# all_zs = np.array(all_zs)
	# all_thetas = np.array(all_thetas)
	return (n, all_zs, all_thetas, all_ps)

def plot_clusters(n, all_zs, all_thetas, all_ps):
	plt.scatter(all_zs, all_thetas, c=all_ps)

result = generate_clusters()
# print(result)
plot_clusters(*result)

(n, all_zs, all_thetas, all_ps) = result

print(all_zs, len(all_zs))
print(all_thetas, len(all_thetas))
print(all_ps, len(all_ps))


#%%


def g(m, Dij):
	"""
	The purpose of this function is to scale the energy levels
	so that the lowest levels are more separated from each other.
	"""

	# print("Dij", Dij)

	# return 1
 
	# w = -math.cos(math.pi * Dij) # close to 1 when Dij is close to 0, close to -1 when Dij is close to 1
	# w += -math.tanh(Dij ** 0.5) * 0.1
	# return w


	# Dij -= 2
	# Dij *= 5
	# print("Dij =", Dij)
	# return math.log(Dij)
	# return Dij
	return 1 + math.log(Dij*m)
	# return Dij ** 0.25 + Dij ** 0.5
	# return m + math.log(Dij)
	# m = 5
	# return 1 - math.exp(-m*Dij)

def create_qubo(Z, T, P, nT, nV, m = 1):
	"""
	Creates a QUBO (Quadratic Unconstrained Binary Optimization) matrix based on the given parameters.

	Args:
		Z: A list of track z-positions.
		deltaZ: A list of track z-position uncertainties.
		nT (int): The number of tracks.
		nV (int): The number of vertices.
		m (float): The parameter used in the distance calculation.

	Returns:
		defaultdict: The QUBO matrix representing the optimization problem.

	Equation:
		Q_p = ∑ₖⁿᵥ ∑ᵢⁿₜ ∑ⱼⁿₜ₍ᵢ₎ pᵢₖ pⱼₖ g(D(ᵢ, ⱼ); m) 
			  + λ ∑ᵢⁿₜ (1 - ∑ₖⁿᵥ pᵢₖ)²

	QUBO Terms:
		- ∑ₖⁿᵥ ∑ᵢⁿₜ ∑ⱼⁿₜ₍ᵢ₎ pᵢₖ pⱼₖ g(D(ᵢ, ⱼ); m): Represents the pairwise interaction term between tracks and vertices, weighted by the distance function.
		- λ ∑ᵢⁿₜ (1 - ∑ₖⁿᵥ pᵢₖ)²: Represents the constraint term penalizing the absence of tracks in vertices.

	Note: The QUBO matrix is represented as a defaultdict with default value 0. The non-zero elements represent the QUBO terms.
	Reference: page 3, http://web3.arxiv.org/pdf/1903.08879.pdf
	"""

	qubo = defaultdict(float)
	Dij_max = 0

	# Define QUBO terms for the first summation
	for k in range(nV):
		for i in range(nT):
			for j in range(i+1, nT):
				# Dij = abs(Z[i] - Z[j]) / (deltaZ[i]**2 + deltaZ[j]**2)**.5
				# Dij_max = max(Dij_max, Dij)
	
				# modulo 1.0 so we can compare angles (which have been normalized to [0, 1])
				Dij = ((Z[i] - Z[j])**2 + (angle_diff(T[i], T[j]) % 1.0)**2) ** 0.5
				Dij_max = max(Dij_max, Dij)
				qubo[(i+nT*k, j+nT*k)] = g(m, Dij) # * (P[i] + P[j]) # prevent high momentum tracks from being assigned to same vertex

	print("Dij_max", Dij_max, "Max before constraint", get_max_coeff(qubo))
	# lam = 1.2 * Dij_max
	# lam = 1.0 * g(m, Dij_max) # was 1.0
	lam = 1.0 * get_max_coeff(qubo)

	# Define QUBO terms for penalty summation
	# Note, we ignore a constant 1 as it does not affect the optimization
	for i in range(nT):
		for k in range(nV):
			qubo[(i+nT*k, i+nT*k)] -= lam
			for l in range(k+1, nV):
				qubo[(i+nT*k, i+nT*l)] += 2 * lam

	return qubo

def get_max_coeff(mydict):
	return max([abs(v) for v in mydict.values()])

def angle_diff(a, b):
	return 2*abs((a - b + 0.5) % 1.0 - 0.5)

# %%

# def create_bqm(Z, T, P, nT, nV, m = 1):
# 	max_dist = get_max_distance(Z, T)
# 	csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)
# 	csp.add_constraint()
# 	pass

# def get_distance(Z_i, T_i, Z_j, T_j):
# 	return ((Z_i - Z_j)**2 + ((T_i - T_j) % 1.0)**2) ** 0.5

# def get_max_distance(Z, T):
# 	max_distance = 0
# 	for i in range(len(Z)):
# 		for j in range(i+1, len(Z)):
# 			max_distance = max(max_distance, get_distance(Z[i], T[i], Z[j], T[j]))
# 	return max_distance

# %%

# one dimensional solution plotter.
def plot_solution(Z, nT, nV, solution):
	print(Z)
	plt.figure()
	# plt.hlines(1, min(Z),max(Z))  # Draw a horizontal line

	palette = ['b', 'g', 'r', 'c', 'm', 'y'] # 6 is enough

	print(solution)

	track_to_vertex = [None] * nT

	for num, bool in solution.items():
		if bool == 1:
			i = num % nT # track number
			k = num // nT # vertex number

			if track_to_vertex[i] != None:
				print("Invalid solution! Track assigned to multiple vertices.")
				return
			else:
				track_to_vertex[i] = k

	# print(track_to_vertex)
	
	if None in track_to_vertex:
		print("Invalid solution! Track assigned to no vertex :(")
		return

	vertex_to_Zs = [[] for _ in range(nV)]
	for track, vertex in enumerate(track_to_vertex):
		vertex_to_Zs[vertex].append(Z[track])

	print(vertex_to_Zs)

	# print(len(Z), len(colorlist))

	plt.eventplot(vertex_to_Zs, orientation='horizontal', colors=palette[:nV], linewidths=1)
	# plt.axis('off')
	plt.yticks([])
	plt.show()

def plot_solution2(Z, T, P, nT, nV, solution):

	palette = ['b', 'g', 'r', 'c', 'm', 'y'] # 6 is enough

	print(solution)

	track_to_vertex = [None] * nT

	for num, bool in solution.items():
		if bool == 1:
			i = num % nT # track number
			k = num // nT # vertex number

			if track_to_vertex[i] != None:
				print("Invalid solution! Track assigned to multiple vertices.")
				return
			else:
				track_to_vertex[i] = k

	# print(track_to_vertex)
	
	if None in track_to_vertex:
		print("Invalid solution! Track assigned to no vertex :(")
		return

	vertex_to_Zs = [[] for _ in range(nV)]
	vertex_to_Thetas = [[] for _ in range(nV)]
	vertex_to_Ps = [[] for _ in range(nV)]
	for track, vertex in enumerate(track_to_vertex):
		vertex_to_Zs[vertex].append(Z[track])
		vertex_to_Thetas[vertex].append(T[track])
		vertex_to_Ps[vertex].append(P[track])

	print(vertex_to_Zs)
	print(vertex_to_Thetas)
	print(vertex_to_Ps)
	vertex_to_Ps = [[40 * p ** 0.5 for p in ps] for ps in vertex_to_Ps]
	plt.figure()
	for i in range(nV):
		plt.scatter(vertex_to_Zs[i], vertex_to_Thetas[i], c=palette[i], s=vertex_to_Ps[i], alpha=0.5)
 


# %%

if __name__ == "__main__":
	# nT = 16
	# nV = 4
	# EVT = 9
	# data_file = f'../clustering_data/{nV}Vertices_{nT}Tracks_100Samples/{nV}Vertices_{nT}Tracks_Event{EVT}/serializedEvents.json'
	
	# Z = []
	# deltaZ = []

	# with open(data_file, 'r') as inputFile:
	# 	for primary_vertex, tracks in json.load(inputFile):
	# 		for z, delta_z in tracks:
	# 			Z.append(z)
	# 			deltaZ.append(delta_z)

	# qubo = create_qubo(Z, deltaZ, 
	# 				nT, nV, m = 0.5/(nV**0.5))
 
	nT = len(all_zs)
	nV = n
	Z = all_zs
	T = all_thetas
	P = all_ps

	qubo = create_qubo(Z, T, P, nT, nV, m=nV-1)

	print(qubo)
	strength = math.ceil(get_max_coeff(qubo))

	print("Max strength", strength)

	# print("Z =", Z)
	# print("T =", T)
	# print("P =", P)
	# print("nV =", nV)
	# print("nT =", nT)
	# print("m =", m)

	# print(nT, nV, m)


	#sampler = LeapHybridSampler(token='DEV-0c064bac1884ffbe99c32c0c572a9390eb918320')
	sampler = EmbeddingComposite(DWaveSampler(token='DEV-0c064bac1884ffbe99c32c0c572a9390eb918320'))
	# response = sampler.sample_qubo(qubo, num_reads=50, chain_strength=1200, annealing_time = 2000)


	response = sampler.sample_qubo(qubo, num_reads=100, chain_strength=strength, annealing_time = 50)


	# Show the problem in inspector, to see chain lengths and solution distribution
	dwave.inspector.show(response)

	print(response)

	idx = 0

	for sample in response:
		idx += 1
		# print("Best Solution:")
  		# plot_solution2(Z, T, P, nT, nV, sample)
		print(sample)
		# if idx > 5:
		# 	break
		break  # Exit after printing the first sample

	best = response.first.sample
	print(best)
	plot_solution2(Z, T, P, nT, nV, best)
	

# %%
