import json
import dimod
import math
from collections import defaultdict
from dwave.system import LeapHybridSampler, DWaveSampler, EmbeddingComposite
import dwave.inspector
# this is to import and process simulated data
inputFile = open('cluster_data/2Vertices_10Tracks_100Samples/2Vertices_10Tracks_Event10/serializedEvents.json')
d_vertextracks = json.load(inputFile)
zT_i = []
zT_unc_i = []
for vertexTracks in d_vertextracks:
  primary_vertex = vertexTracks[0]
  tracks = vertexTracks[1]
  for i in tracks:
    zT_i.append(i[0])
    zT_unc_i.append(i[1])
    #print(str(i[0])+" "+str(i[1])) # i[0] is position, i[1] is uncertainty
# function
# Function to create QUBO with the sum of binary variables
def create_qubo(num_tracks, num_vertices, m):
    qubo = defaultdict(float)
    L=0
    distance_list = []
    # Define QUBO terms for the first summation
    for k in range(num_vertices):
      for i in range(num_tracks):
        for j in range(i, num_tracks):
          distance = abs(zT_i[i] - zT_i[j]) / (zT_unc_i[i]**2 + zT_unc_i[j]**2)**.5
          distance_list.append(distance)
          qubo[(i+num_tracks*k, j+num_tracks*k)] = 1-math.exp(-m*(distance))
          
    # from paper, lambda has a specific value
    L = 1.2*max(distance_list)
    # Define QUBO terms for penalty summation
    for i in range(num_tracks):
      # Nothing to do here
      for k in range(num_vertices):
        # for both products of -pik*1
        qubo[(i+num_tracks*k,i+num_tracks*k)] -= 2*L
        for l in range(k, num_vertices):
          # for both products of -pik*-pij
          qubo[(i+num_tracks*k, i+num_tracks*l)] += 2*L
    return qubo
qubo = create_qubo(10, 2, 5)
print(qubo)
#quit()
#sampler = LeapHybridSampler(token='DEV-0c064bac1884ffbe99c32c0c572a9390eb918320')
sampler = EmbeddingComposite(DWaveSampler(token='DEV-0c064bac1884ffbe99c32c0c572a9390eb918320'))
response = sampler.sample_qubo(qubo, num_reads=50, chain_strength=1.2, annealing_time = 50)
# Show the problem in inspector, to see chain lengths and solution distribution
dwave.inspector.show(response)
# Analyze the response
for sample in response:
    print("Best Solution:")
    print(sample)
    break  # Exit after printing the first sample
