#%% Python 3.10

import json
import dimod
import math
from collections import defaultdict
from dwave.system import LeapHybridSampler, DWaveSampler, EmbeddingComposite
import dwave.inspector

#%%


def g(m, Dij):
	"""
	The purpose of this function is to scale the energy levels
	so that the lowest levels are more separated from each other.
	"""
	return 1 - math.exp(-m*Dij)

def create_qubo(Z, deltaZ, nT, nV, m = 5):
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
			for j in range(i, nT):
				Dij = abs(Z[i] - Z[j]) / (deltaZ[i]**2 + deltaZ[j]**2)**.5
				Dij_max = max(Dij_max, Dij)
				qubo[(i+nT*k, j+nT*k)] = g(m, Dij) #q(ik, jk)

	lam = 1.2 * Dij_max

	# Define QUBO terms for penalty summation
	# Note, we ignore a constant 1 as it does not affect the optimization
	for i in range(nT):
		for k in range(nV):
			qubo[(i+nT*k, i+nT*k)] -= lam
			for l in range(k, nV):
				qubo[(i+nT*k, i+nT*l)] += lam

	return qubo

# %%

if __name__ == "__main__":
	nT = 10
	nV = 2
	EVT = 9
	data_file = f'cluster_data/PrimaryVertexingArtificialData/PrimaryVertexingArtificialData/{nV}Vertices_{nT}Tracks_100Samples/{nV}Vertices_{nT}Tracks_Event{EVT}/serializedEvents.json'
	
	Z = []
	deltaZ = []

	with open(data_file, 'r') as inputFile:
		for primary_vertex, tracks in json.load(inputFile):
			for z, delta_z in tracks:
				Z.append(z)
				deltaZ.append(delta_z)


	qubo = create_qubo(Z, deltaZ, 
					nT, nV, m = 5)
	print(qubo)


	#sampler = LeapHybridSampler(token='DEV-0c064bac1884ffbe99c32c0c572a9390eb918320')
	sampler = EmbeddingComposite(DWaveSampler(token='DEV-0c064bac1884ffbe99c32c0c572a9390eb918320'))
	response = sampler.sample_qubo(qubo, num_reads=50, chain_strength=1200, annealing_time = 2000)

	# Show the problem in inspector, to see chain lengths and solution distribution
	dwave.inspector.show(response)

	for sample in response:
		print("Best Solution:")
		print(sample)
		break  # Exit after printing the first sample
