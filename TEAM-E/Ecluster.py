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
	num_tracks = 20

	stdev = 0.05

	p = np.random.rand(num_tracks)
	p = 1/(p**2 + 0.001) # 3 orders of magnitude diff between min and max. corresponds to: 30 MeV, 30 GeV scaled
	p /= p.max()
	print(p)
	# n = sum(p > 0.3)
	# print(n)
	thresh = get_kt_dij_avg(num_tracks, p)
	# print(thresh, "kt_dij_avg")

	Dib = 1/(p**2)
	# print("inverse square momentum", Dib)
	n = sum(Dib > thresh)

	print("n", n)

	if n > 5 or n < 2:
	# if n != 3: # just for now
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
	truth = [] # which cluster each point belongs to
	for i in range(n):
		z = vertex_zs[i]
		theta = vertex_thetas[i]
		for j in range(points_per_cluster):
			all_zs.append(z + np.random.normal(scale=stdev))
			all_thetas.append((theta + np.random.normal(scale=stdev)) % 1.0) # modulo 1.0 to keep it in the range
			all_ps.append(get_rand_from(soft_ps))
			truth.append(i)
		all_zs.append(z)
		all_thetas.append(theta)
		all_ps.append(get_rand_from(hard_ps))
		truth.append(i)
	# all_zs = np.array(all_zs)
	# all_thetas = np.array(all_thetas)
	return (n, all_zs, all_thetas, all_ps, truth)

def plot_clusters(n, all_zs, all_thetas, all_ps, truth):
	palette = ['b', 'g', 'r', 'c', 'm', 'y'] # 6 is enough
	colors = [palette[i] for i in truth]
	scaled = get_reasonable_sizes_for_plotting_momentum(all_ps)
	# plt.scatter(all_zs, all_thetas, c=all_ps)
	plt.figure()
	plt.title("Generated Clusters")
	plt.xlabel("Z (normalized)")
	plt.ylabel("θ (normalized)")
	plt.grid()
	for (i, z) in enumerate(all_zs):
		plt.scatter(z, all_thetas[i], c=colors[i], s=scaled[i], alpha=0.5)


result = generate_clusters()
# print(result)
plot_clusters(*result)

(n, all_zs, all_thetas, all_ps, truth) = result

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

	qubo = defaultdict(float)
	Dij_max = 0

	# Define QUBO terms for the first summation
	for k in range(nV):
		for i in range(nT):
			for j in range(i+1, nT):
				Dij = ((Z[i] - Z[j])**2 + angle_diff(T[i], T[j])**2) ** 0.5
				Dij_max = max(Dij_max, Dij)
				# print(g(m, Dij), min(P[i], P[j]))
				# we add the min(P[i], P[j]) term to prevent high momentum tracks from being assigned to the same vertex
				qubo[(i+nT*k, j+nT*k)] = g(m, Dij) + min(P[i], P[j]) # * (P[i] + P[j]) # prevent high momentum tracks from being assigned to same vertex

	print("Dij_max", Dij_max, "Max before constraint", get_max_coeff(qubo))
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
	return 2*abs((a - b + 0.5) % 1.0 - 0.5) # mult by 2 to make it [0, 1] instead of [0, 0.5]

def get_reasonable_sizes_for_plotting_momentum(P):
	return [500 * p for p in P]
	# return [50 * p ** 0.25 for p in P]

# refer to robin's new plan pdf
def get_kt_dij_avg(nT, P, R=1.0):
	A = 1 # 1 because both axes are normalized
	median_p = sorted(P)[nT//2]
	return nT / (A * R ** 2) * (median_p ** -2)

# %%

def plot_solution2(Z, T, P, nT, nV, solution):

	palette = ['b', 'g', 'r', 'c', 'm', 'y'] # 6 is enough

	# print(solution)

	vertex_to_Zs, vertex_to_Thetas, vertex_to_Ps = interpret_solution(Z, T, P, nT, nV, solution)

	if vertex_to_Zs == None: # invalid solution
		return

	print(vertex_to_Zs)
	print(vertex_to_Thetas)
	print(vertex_to_Ps)
	vertex_to_Ps = [get_reasonable_sizes_for_plotting_momentum(ps) for ps in vertex_to_Ps]
	plt.figure()
	plt.title("Annealer solution")
	plt.xlabel("Z (normalized)")
	plt.ylabel("θ (normalized)")
	plt.grid()
	for i in range(nV):
		plt.scatter(vertex_to_Zs[i], vertex_to_Thetas[i], c=palette[i], s=vertex_to_Ps[i], alpha=0.5)
 
# %%

def interpret_solution(Z, T, P, nT, nV, solution):
	track_to_vertex = [None] * nT

	for num, bool in solution.items():
		if bool == 1:
			i = num % nT # track number
			k = num // nT # vertex number

			if track_to_vertex[i] != None:
				print("Invalid solution! Track assigned to multiple vertices.")
				return None, None, None
			else:
				track_to_vertex[i] = k
	
	if None in track_to_vertex:
		print("Invalid solution! Track assigned to no vertex :(")
		return None, None, None

	vertex_to_Zs = [[] for _ in range(nV)]
	vertex_to_Thetas = [[] for _ in range(nV)]
	vertex_to_Ps = [[] for _ in range(nV)]
	for track, vertex in enumerate(track_to_vertex):
		vertex_to_Zs[vertex].append(Z[track])
		vertex_to_Thetas[vertex].append(T[track])
		vertex_to_Ps[vertex].append(P[track])

	return vertex_to_Zs, vertex_to_Thetas, vertex_to_Ps

# %%

if __name__ == "__main__":
 
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

# Graveyard for relics of the past:
	# """
	# Creates a QUBO (Quadratic Unconstrained Binary Optimization) matrix based on the given parameters.

	# Args:
	# 	Z: A list of track z-positions.
	# 	deltaZ: A list of track z-position uncertainties.
	# 	nT (int): The number of tracks.
	# 	nV (int): The number of vertices.
	# 	m (float): The parameter used in the distance calculation.

	# Returns:
	# 	defaultdict: The QUBO matrix representing the optimization problem.

	# Equation:
	# 	Q_p = ∑ₖⁿᵥ ∑ᵢⁿₜ ∑ⱼⁿₜ₍ᵢ₎ pᵢₖ pⱼₖ g(D(ᵢ, ⱼ); m) 
	# 		  + λ ∑ᵢⁿₜ (1 - ∑ₖⁿᵥ pᵢₖ)²

	# QUBO Terms:
	# 	- ∑ₖⁿᵥ ∑ᵢⁿₜ ∑ⱼⁿₜ₍ᵢ₎ pᵢₖ pⱼₖ g(D(ᵢ, ⱼ); m): Represents the pairwise interaction term between tracks and vertices, weighted by the distance function.
	# 	- λ ∑ᵢⁿₜ (1 - ∑ₖⁿᵥ pᵢₖ)²: Represents the constraint term penalizing the absence of tracks in vertices.

	# Note: The QUBO matrix is represented as a defaultdict with default value 0. The non-zero elements represent the QUBO terms.
	# Reference: page 3, http://web3.arxiv.org/pdf/1903.08879.pdf
	# """

	# # one dimensional solution plotter.
# def plot_solution(Z, nT, nV, solution):
# 	print(Z)
# 	plt.figure()
# 	# plt.hlines(1, min(Z),max(Z))  # Draw a horizontal line

# 	palette = ['b', 'g', 'r', 'c', 'm', 'y'] # 6 is enough

# 	print(solution)

# 	track_to_vertex = [None] * nT

# 	for num, bool in solution.items():
# 		if bool == 1:
# 			i = num % nT # track number
# 			k = num // nT # vertex number

# 			if track_to_vertex[i] != None:
# 				print("Invalid solution! Track assigned to multiple vertices.")
# 				return
# 			else:
# 				track_to_vertex[i] = k

# 	# print(track_to_vertex)
	
# 	if None in track_to_vertex:
# 		print("Invalid solution! Track assigned to no vertex :(")
# 		return

# 	vertex_to_Zs = [[] for _ in range(nV)]
# 	for track, vertex in enumerate(track_to_vertex):
# 		vertex_to_Zs[vertex].append(Z[track])

# 	print(vertex_to_Zs)

# 	# print(len(Z), len(colorlist))

# 	plt.eventplot(vertex_to_Zs, orientation='horizontal', colors=palette[:nV], linewidths=1)
# 	# plt.axis('off')
# 	plt.yticks([])
# 	plt.show()

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