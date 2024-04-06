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
from sklearn.metrics.cluster import rand_score
import pandas as pd

#%%

def get_rand_from(v):
	elem = v[np.random.randint(len(v))]
	v.remove(elem)
	return elem

def get_nv_from_p(p):
	thresh = get_kt_dij_avg(p, R=1.3)
	Dib = 1/(p**2)
	n = sum(Dib > thresh)
	return n

# refer to robin's new plan pdf
def get_kt_dij_avg(P, R=1.0):
	A = 1 # 1 because both axes are normalized
	nT = len(P)
	median_p = sorted(P)[nT//2]
	return nT / (A * R ** 2) * (median_p ** -2)

# truth and solution should be of form [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
def score_clusters(truth, solution):
	if len(solution) == 0:
		return 0.0
	if solution.isnull().any():
		return 0.0
	return 100.0 * rand_score(truth, solution)
	

# maybe base this on https://arxiv.org/pdf/1405.6569.pdf in the future
def generate_clusters(nt=16, nv=None, std=0.03):

	z_range = 1 # 0 to 1
	theta_range = 1 # 0 to 1 (we're acting like we squished the ranges)
	num_tracks = nt

	p = np.random.rand(num_tracks)
	p = 1/(p**2 + 0.001) # 3 orders of magnitude diff between min and max. corresponds to: 30 MeV, 30 GeV scaled
	p /= p.max()
	n = get_nv_from_p(p)

	if nv is None:
		if n > 5 or n < 2:
			return generate_clusters(nt, None, std)
	else:
		if n != nv:
			return generate_clusters(nt, nv, std)

	p = np.sort(p)[::-1].tolist()

	hard_ps = p[:n]
	soft_ps = p[n:]

	vertex_zs = np.random.rand(n) * z_range
	vertex_thetas = np.random.rand(n) * theta_range

	points_per_cluster = num_tracks // n
	remainder = num_tracks % n

	all_zs = []
	all_thetas = []
	all_ps = []
	truth = [] # which cluster each point belongs to
	for i in range(n):
		z = vertex_zs[i]
		theta = vertex_thetas[i]

		# -1 because we're going to add the vertex itself too
		toadd = points_per_cluster - 1 + (1 if i < remainder else 0)
		for j in range(toadd):
			all_zs.append(z + np.random.normal(scale=std))
			all_thetas.append((theta + np.random.normal(scale=std)) % 1.0) # modulo 1.0 to keep it in the range
			all_ps.append(get_rand_from(soft_ps))
			truth.append(i)
		all_zs.append(z)
		all_thetas.append(theta)
		all_ps.append(get_rand_from(hard_ps))
		truth.append(i)
	# all_zs = np.array(all_zs)
	# all_thetas = np.array(all_thetas)
	return pd.DataFrame({'z': all_zs, 'theta': all_thetas, 'momentum': all_ps, 'truegroup': truth})
	# return (n, all_zs, all_thetas, all_ps, truth)

def get_reasonable_sizes_for_plotting_momentum(P):
	return [500 * p for p in P]
	# return [50 * p ** 0.25 for p in P]

def plot_clusters(df, grouping, title):
	all_zs = df['z']
	all_thetas = df['theta']
	all_ps = df['momentum']

	palette = ['b', 'g', 'r', 'c', 'm', 'y'] # 6 is enough
	colors = [palette[i] for i in grouping]
	scaled = get_reasonable_sizes_for_plotting_momentum(all_ps)
	# plt.scatter(all_zs, all_thetas, c=all_ps)
	plt.figure()
	plt.title(title)
	plt.xlabel("Z (normalized)")
	plt.ylabel("θ (normalized)")
	plt.grid()
	for (i, z) in enumerate(all_zs):
		plt.scatter(z, all_thetas[i], c=colors[i], s=scaled[i], alpha=0.5)


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

def create_qubo(df, m = 1):
	Z = df['z']
	T = df['theta']
	P = df['momentum']
	nV = get_nv_from_p(P)
	nT = len(Z)

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

# %%

# returns vertex_to_Zs, vertex_to_Thetas, vertex_to_Ps, track_to_vertex
def set_solution_from_annealer_response(df, response):
	nT = len(df)
	track_to_vertex = [None] * nT
	# nV = get_nv_from_p(df['momentum'])

	for num, bool in response.items():
		if bool == 1:
			i = num % nT # track number
			k = num // nT # vertex number

			if track_to_vertex[i] != None:
				print("Invalid solution! Track assigned to multiple vertices.")
				# track_to_vertex = None
				df['qagroup'] = None
				return
			else:
				track_to_vertex[i] = k
	
	if None in track_to_vertex:
		print("Invalid solution! Track assigned to no vertex :(")
		track_to_vertex = None
	
	df['qagroup'] = track_to_vertex
	# df['qagroup'] = track_to_vertex

# %%

if __name__ == "__main__":
	# keep in mind that a large number of qubits results in chain breaks. 
	# empirically 60-64 qubits is pretty much the max you want to use
 	# which means nt * nv <= 60

	# df = generate_clusters(nv=None)
	df = generate_clusters(nt=30, nv=2, std=0.05)
	# df = generate_clusters(nt=18, nv=3, std=0.03)
	# df = generate_clusters(nt=15, nv=4, std=0.03)
	# df = generate_clusters(nt=12, nv=5, std=0.03)
	plot_clusters(df, df['truegroup'], "Generated Clusters")
	print(df)

	nv = get_nv_from_p(df['momentum'])
	nt = len(df)

	qubo = create_qubo(df, m=nv-1)

	# print(qubo)

	strength = math.ceil(get_max_coeff(qubo))

	print("Max strength", strength)

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

	set_solution_from_annealer_response(df, best)
	print(df)
	score = score_clusters(df['truegroup'], df['qagroup'])
	if not df['qagroup'].isnull().any():
		plot_clusters(df, df['qagroup'], f"Annealer solution, Rand index {score:.1f}%")
	print("Score:", score)
	

#%%

# R^2 is supposedly 1/pi (?)
R_squared = None
group_list = []
print(df) # just checked to make sure data was in the datagram
df_momentum = df.sort_values(by='momentum', ascending=False) # sorted values in the datafram
x = len(df_momentum['momentum']) # finding N value
x_range = range(1, x+1) # range is x+1 since range doesnt include last number
y = df_momentum['momentum'] # making array of y values
# plt.bar(x_range, y) # plotting all y momentum values
#plt.show()
# values for inverse squared plot
y_inverse_square = ((df_momentum['momentum']) ** (-2))
plt.bar(x_range, y_inverse_square) # plotting the inverse square graph
plt.show()

# area = np.trapz(y_inverse_square, x_range) # finding the area under the curve using numpys method
# # print(f"area: {area}")
# N = 66 # setting our N value = 66
# delta_avg2 = (area/N) # calculating delta avg squared
# print(delta_avg2)
# k1med = np.median(y)
# print(k1med)
# avg_dij = np.pi*k1med*delta_avg2 # calculating average dij with out prediction for R^2=1/pi
# print(avg_dij, "avg_dij")

# returns the distance between two points in cylindrical coordinates
# assumes rho is 1
def delta_distance(point_i, point_j):
    return (angle_diff(point_i['theta'], point_j['theta']) ** 2 + ((point_i['z'] - point_j['z']) ** 2)) ** 0.5
            
def calculate_d_ij(particle_i, particle_j):
    k_ti = particle_i['momentum']
    k_tj = particle_j['momentum']
    delta_ij = delta_distance(particle_i, particle_j)
    d_ij = min(k_ti ** (-2), k_tj ** (-2)) * ((delta_ij ** (2)) / (R_squared))
    return d_ij
# Function to sort through groupings of particles according to d_ij
def sort_pairs_by_d_ij(particles_df):
    # Create a list of all possible pairs of particles
    pairs = [(i, j) for i in range(len(particles_df)) for j in range(i+1, len(particles_df))]
    # Sort pairs by d_ij
    pairs.sort(key=lambda x: calculate_d_ij(df.iloc[x[0]], df.iloc[x[1]]))
    return pairs

def anti_kt(particles_df):
    # initialize groups (each entry assigned an initial group containing only that element)
    particles_df['ktgroup'] = 0
    for index, _ in particles_df.iterrows():
        particles_df.at[index, 'ktgroup'] = index
    # sort particle groupings by d_ij
    pairs = sort_pairs_by_d_ij(df)
    # iterate through sorted pairs
    for i, j in pairs:
        particle_i = df.iloc[i]
        particle_j = df.iloc[j]
        k_ti = particle_i['momentum']
        d_ij = calculate_d_ij(particle_i, particle_j)
        d_iB = k_ti ** (-2)
        #print(f"d_ij: {d_ij}")
        #print(f"d_iB: {d_iB}")
        # Combine when d_ij < d_iB; stop when d_ij >= d_iB (note: not specified which inequality should be inclusive)
        if (d_ij < d_iB):
            combine_groups(df, particle_i['ktgroup'], particle_j['ktgroup'])
        else:
            print("STOP")
            break
def combine_groups(df, group_a, group_b):
    group_combined = min(group_a, group_b)
    group_list.append(group_a)
    group_list.append(group_b)
    for index, row in df.iterrows():
        if (row['ktgroup'] == group_a or row['ktgroup'] == group_b):
            df.at[index, 'ktgroup'] = group_combined
# option to test values of R_squared for performance
R_squared_range = np.linspace(1, 0.1, 21)
scores = []
for val in R_squared_range:
    R_squared = val
    data = df
    anti_kt(data)
    num_groups = len(np.unique(data['ktgroup']))
    scores.append(num_groups)
print(scores)
# output for the scores: [27, 27, 27, 27, 27, 27, 27, 343, 343, 343, 512, 512, 512, 1331, 1331, 1331, 4096, 4096, 4913, 10648, 21952]
# clearly larger values of R_squared give better scores (lower difference from the truth) but only to some point
#quit()
# used to plot specific runs
R_squared = 0.1
anti_kt(df)
fig, ax = plt.subplots()
z = np.array(df['z'])
theta = np.array(df['theta'])
c=0
pmax = max(np.array(df['momentum'])) # used for coloring points based on hardness
for p in df['momentum']:
    greyscale = (1-float(p)/pmax)
    ax.scatter(z[c], theta[c], c=(greyscale, greyscale, greyscale), edgecolors='red') # harder particles should be darker
    c += 1
for i, txt in enumerate(np.array(df['ktgroup'])):
    ax.annotate(txt, (z[i], theta[i]))
groups = np.array(group_list)
groups = np.unique(groups)
print(groups)
plt.show()
#combine_groups(df, 2, 3)
# %%
