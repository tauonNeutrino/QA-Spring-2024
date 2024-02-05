# %%
import json
import numpy as np
from matplotlib import pyplot as plt
nT = 15
nV = 5
EVTS = range(1, 101)

primaries = []
Z = []

for EVT in EVTS:
    data_file = f'clustering_data/{nV}Vertices_{nT}Tracks_100Samples/{nV}Vertices_{nT}Tracks_Event{EVT}/serializedEvents.json'	
    with open(data_file, 'r') as inputFile:
        for primary_vertex, tracks in json.load(inputFile):
            primaries.append(primary_vertex)
            for z, delta_z in tracks:
                Z.append(z)

# %%
# plot histogram for primaries and for Z

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].hist(primaries, bins=30)
ax[0].set_title('Primary Vertices')
ax[0].set_xlabel('z')
ax[0].set_ylabel('Frequency')

ax[1].hist(Z, bins=30)
ax[1].set_title('Tracks')
ax[1].set_xlabel('z')
ax[1].set_ylabel('Frequency')

plt.show()

# %%
# plot two parallel horizontal lines, vertex on first line, track on second line,
# and a line connecting them

fig, ax = plt.subplots(1, 1, figsize=(10, 1))
for _, EVT in zip(range(2), EVTS):
    data_file = f'cluster_data/PrimaryVertexingArtificialData/PrimaryVertexingArtificialData/{nV}Vertices_{nT}Tracks_100Samples/{nV}Vertices_{nT}Tracks_Event{EVT}/serializedEvents.json'	
    with open(data_file, 'r') as inputFile:
        for primary_vertex, tracks in json.load(inputFile):
            # generate a color
            color = (np.random.rand(), np.random.rand(), np.random.rand())
            for z, delta_z in tracks:
                ax.plot([primary_vertex, z], [1, 0], 'o-', color=color)
ax.set_title('Primary Vertices and Tracks')
ax.legend()
plt.show()
# %%
