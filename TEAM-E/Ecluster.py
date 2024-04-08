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
from sklearn.metrics.cluster import rand_score, adjusted_rand_score
import pandas as pd
import random
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os
import mplcyberpunk
# plt.style.use("dark_background")
# plt.style.use("https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle")
plt.style.use("cyberpunk")
# plt.rcParams["font.family"] = "Helvetica"
plt.rcParams['figure.dpi']=300

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
	return 100.0 * adjusted_rand_score(truth, solution)
	

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
	return [500 * p**0.6 for p in P]
	# return [50 * p ** 0.25 for p in P]

def rand_cmap(nlabels, type='bright'):
	from matplotlib.colors import LinearSegmentedColormap
	import colorsys
	import numpy as np

	# Generate color map for bright colors, based on hsv
	if type == 'bright':
		randHSVcolors = [((np.random.uniform(low=0.0, high=1)),
						  np.random.uniform(low=0.7, high=1),
						  np.random.uniform(low=0.9, high=1)) for _ in range(nlabels)]

		# Convert HSV list to RGB
		randRGBcolors = []
		for HSVcolor in randHSVcolors:
			randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

		# random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)
		return randRGBcolors
	# Generate soft pastel colors
	if type == 'soft':
		low = 0.6
		high = 0.95
		randRGBcolors = [(np.random.uniform(low=low, high=high),
						  np.random.uniform(low=low, high=high),
						  np.random.uniform(low=low, high=high)) for _ in range(nlabels)]
		# random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)
		return randRGBcolors

# palette = list(mcolors.XKCD_COLORS)
# palette = cm.tab20(range(20))
# hsv = plt.get_cmap('hsv')
# palette = hsv(np.linspace(0, 1.0, 30))
palette = rand_cmap(30, type='bright')
# random.shuffle(palette)

def plot_clusters(df, grouping, title, saveto=None):
	all_zs = df['z']
	all_thetas = df['theta']
	all_ps = df['momentum']

	# palette = ['b', 'g', 'r', 'c', 'm', 'y'] # 6 is enough
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
		# mplcyberpunk.make_scatter_glow()
	if saveto is not None:
		plt.savefig(saveto)
		plt.close() # avoid displaying the plot

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

group_list = []
# print(df) # just checked to make sure data was in the datagram

# returns the distance between two points in cylindrical coordinates
# assumes rho is 1
def delta_distance(point_i, point_j):
	return (angle_diff(point_i['theta'], point_j['theta']) ** 2 + ((point_i['z'] - point_j['z']) ** 2)) ** 0.5
			
def calculate_d_ij(particle_i, particle_j, R_squared):
	k_ti = particle_i['momentum']
	k_tj = particle_j['momentum']
	delta_ij = delta_distance(particle_i, particle_j)
	d_ij = min(k_ti ** (-2), k_tj ** (-2)) * ((delta_ij ** (2)) / (R_squared))
	return d_ij
# Function to sort through groupings of particles according to d_ij
def sort_pairs_by_d_ij(particles_df, R_squared):
	# Create a list of all possible pairs of particles
	pairs = [(i, j) for i in range(len(particles_df)) for j in range(i+1, len(particles_df))]
	# Sort pairs by d_ij
	pairs.sort(key=lambda x: calculate_d_ij(particles_df.iloc[x[0]], particles_df.iloc[x[1]], R_squared))
	return pairs

def anti_kt(particles_df, R_squared):
	# initialize groups (each entry assigned an initial group containing only that element)
	particles_df['ktgroup'] = 0
	for index, _ in particles_df.iterrows():
		particles_df.at[index, 'ktgroup'] = index
	# sort particle groupings by d_ij
	pairs = sort_pairs_by_d_ij(particles_df, R_squared)
	# iterate through sorted pairs
	for i, j in pairs:
		particle_i = particles_df.iloc[i]
		# print(particle_i)
		particle_j = particles_df.iloc[j]
		# print(particle_j)
		k_ti = particle_i['momentum']
		d_ij = calculate_d_ij(particle_i, particle_j, R_squared)
		d_iB = k_ti ** (-2)
		# print(f"d_ij: {d_ij}")
		# print(f"d_iB: {d_iB}")
		# Combine when d_ij < d_iB; stop when d_ij >= d_iB (note: not specified which inequality should be inclusive)
		if (d_ij < d_iB):
			combine_groups(particles_df, particle_i['ktgroup'], particle_j['ktgroup'])
		else:
			# print("STOP")
			break
def combine_groups(particles_df, group_a, group_b):
	group_combined = min(group_a, group_b)
	group_list.append(group_a)
	group_list.append(group_b)
	for index, row in particles_df.iterrows():
		if (row['ktgroup'] == group_a or row['ktgroup'] == group_b):
			particles_df.at[index, 'ktgroup'] = group_combined


def get_data_config_for(nv, std):
	if nv == 2:
		return 20, 2, std
	if nv == 3:
		return 18, 3, std
	if nv == 4:
		return 16, 4, std
	if nv == 5:
		return 12, 5, std

def create_dataset():
	directory = "../newdata/std03"
	if not os.path.exists(directory):
		os.makedirs(directory)
	
	for i in range(15):
		for nv in range(2, 6):
			nt, nv, std = get_data_config_for(nv, 0.03)
			df = generate_clusters(nt, nv, std)
			df.to_csv(f"{directory}/nv{nv}_{i}.csv", index=False)
	
	directory = "../newdata/std05"
	if not os.path.exists(directory):
		os.makedirs(directory)

	for i in range(10):
		for nv in range(2, 6):
			nt, nv, std = get_data_config_for(nv, 0.05)
			df = generate_clusters(nt, nv, std)
			df.to_csv(f"{directory}/nv{nv}_{i}.csv", index=False)


def solve_file(file, std):
	df = pd.read_csv(file)
	run_qa(df)
	qascore = score_clusters(df['truegroup'], df['qagroup'])
	R_squared = (std*3.0)**2 # R is the radius param so a good guess is 3 * stdev
	anti_kt(df, R_squared)
	ktscore = score_clusters(df['truegroup'], df['ktgroup'])
	df['qascore'] = qascore
	df['ktscore'] = ktscore
	print(df)
	df.to_csv(file, index=False)

def visualize_file(file, out=None):
	outgen = out + "_gen.png" if out is not None else None
	outqa = out + "_qa.png" if out is not None else None
	outkt = out + "_kt.png" if out is not None else None
	df = pd.read_csv(file)
	plot_clusters(df, df['truegroup'], "Generated Clusters", saveto=outgen)
	if not df['qagroup'].isnull().any():
		qascore = score_clusters(df['truegroup'], df['qagroup'])
		plot_clusters(df, df['qagroup'], f"Annealer solution, Adj. Rand index {qascore:.1f}%", saveto=outqa)
	ktscore = score_clusters(df['truegroup'], df['ktgroup'])
	plot_clusters(df, df['ktgroup'], f"Anti-KT solution, Adj. Rand index {ktscore:.1f}%", saveto=outkt)

def run_qa(df):
	nv = get_nv_from_p(df['momentum'])
	qubo = create_qubo(df, m=nv-1)
	strength = math.ceil(get_max_coeff(qubo))
	# print("Max strength", strength)
	#sampler = LeapHybridSampler(token='DEV-0c064bac1884ffbe99c32c0c572a9390eb918320')
	sampler = EmbeddingComposite(DWaveSampler(token='DEV-0c064bac1884ffbe99c32c0c572a9390eb918320'))
	# response = sampler.sample_qubo(qubo, num_reads=50, chain_strength=1200, annealing_time = 2000)
	response = sampler.sample_qubo(qubo, num_reads=100, chain_strength=strength, annealing_time = 50)
	best = response.first.sample
	# print(best)
	set_solution_from_annealer_response(df, best)

# %%


def demo():
	std = 0.03
	# df = generate_clusters(nv=None)
	# df = generate_clusters(nt=25, nv=2, std=std)
	df = generate_clusters(nt=18, nv=3, std=std)
	# df = generate_clusters(nt=16, nv=4, std=std)
	# df = generate_clusters(nt=12, nv=5, std=std)
	plot_clusters(df, df['truegroup'], "Generated Clusters")
	print(df)

	run_qa(df)

	print(df)
	qascore = score_clusters(df['truegroup'], df['qagroup'])
	if not df['qagroup'].isnull().any():
		plot_clusters(df, df['qagroup'], f"Annealer solution, Adj. Rand index {qascore:.1f}%")
	
	# print("Score:", score)

	R_squared = (std*2.5)**2 # R is the radius param so a good guess is 3 * stdev
	anti_kt(df, R_squared)
	ktscore = score_clusters(df['truegroup'], df['ktgroup'])
	plot_clusters(df, df['ktgroup'], f"Anti-KT solution, Adj. Rand index {ktscore:.1f}%", saveto="../newdata/kt.png")
	
	print(df)

	df['qascore'] = qascore
	df['ktscore'] = ktscore

	df.to_csv("../newdata/df.csv", index=False)


#%%
if __name__ == "__main__":
	# demo()
	
	for i in range(10):
		nv = 5
		file = f"../newdata/std05/nv{nv}_{i}.csv"
		solve_file(file, 0.05)
		print(i)
		# visualize_file(file)



# %%


def read_accuracies(std, nv):
	accuraciesqa = []
	accuracieskt = []
	for i in range(10 if std == 0.05 else 15):
		file = f"../newdata/std05/nv{nv}_{i}.csv" if std == 0.05 else f"../newdata/std03/nv{nv}_{i}.csv"
		df = pd.read_csv(file)
		accuraciesqa.append(df['qascore'].mean())
		accuracieskt.append(df['ktscore'].mean())
		# print(i)
	return accuraciesqa, accuracieskt

def plot_accuracies(nv, std):
	accuraciesqa, accuracieskt = read_accuracies(std, nv)
	# accuraciesqa = np.mean(accuraciesqa)
	# accuracieskt = np.mean(accuracieskt)
	plt.hist([accuraciesqa, accuracieskt], label=['Quantum Annealing', 'Anti-Kt Method'])
	# plt.hist(accuraciesqa, label='Quantum Annealing', alpha=0.7)
	# plt.hist(accuracieskt, label='Anti-Kt Method', alpha=0.7)
	plt.xlabel("Adjusted Rand Index")
	plt.ylabel("Frequency")
	plt.title(f"Adjusted Rand Index for {nv} vertices, std={std}")
	plt.legend()
	plt.savefig(f"../hist_figs/hist_nv{nv}_std{std}.png")
	plt.show()
	

def plot_accuracies_all():
	allqa = []
	allkt = []
	for nv in range(2, 6):
		for std in [0.03, 0.05]:
			accuraciesqa, accuracieskt = read_accuracies(std, nv)
			allqa.extend(accuraciesqa)
			allkt.extend(accuracieskt)
	plt.hist([allkt, allqa], label=['Anti-Kt Method', 'Quantum Annealing'])
	plt.xlabel("Adjusted Rand Index")
	plt.ylabel("Frequency")
	plt.title(f"Adjusted Rand Index for all numbers of vertices, all std")
	plt.legend()
	plt.savefig(f"../hist_figs/hist_all.png")
	plt.show()

def plot_accuracies_with_std(std):
	allqa = []
	allkt = []
	for nv in range(2, 6):
		accuraciesqa, accuracieskt = read_accuracies(std, nv)
		allqa.extend(accuraciesqa)
		allkt.extend(accuracieskt)
	plt.hist([allkt, allqa], label=['Anti-Kt Method', 'Quantum Annealing'])
	plt.xlabel("Adjusted Rand Index")
	plt.ylabel("Frequency")
	plt.title(f"Adjusted Rand Index for all numbers of vertices, std={std}")
	plt.legend()
	plt.savefig(f"../hist_figs/hist_std{std}.png")
	plt.show()

def plot_accuracies_with_nv(nv):
	allqa = []
	allkt = []
	for std in [0.03, 0.05]:
		accuraciesqa, accuracieskt = read_accuracies(std, nv)
		allqa.extend(accuraciesqa)
		allkt.extend(accuracieskt)
	plt.hist([allkt, allqa], label=['Anti-Kt Method', 'Quantum Annealing'])
	plt.xlabel("Adjusted Rand Index")
	plt.ylabel("Frequency")
	plt.title(f"Adjusted Rand Index for all std, nv={nv}")
	plt.legend()
	plt.savefig(f"../hist_figs/hist_nv{nv}.png")
	plt.show()

# plot_accuracies(0.05, 2)
plot_accuracies_all()

plot_accuracies_with_std(0.03)
plot_accuracies_with_std(0.05)

plot_accuracies_with_nv(2)
plot_accuracies_with_nv(3)
plot_accuracies_with_nv(4)
plot_accuracies_with_nv(5)

plot_accuracies(2, 0.03)
plot_accuracies(2, 0.05)

plot_accuracies(3, 0.03)
plot_accuracies(3, 0.05)

plot_accuracies(4, 0.03)
plot_accuracies(4, 0.05)

plot_accuracies(5, 0.03)
plot_accuracies(5, 0.05)
# %%
