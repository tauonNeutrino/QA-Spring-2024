import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import *
import os

current_directory = os.getcwd()

file_name = 'Downloads/3D_randgen.csv'

full_path = os.path.join(current_directory, file_name)

try:
    df = pd.read_csv(file_name, header=0, index_col=0, )
except FileNotFoundError:
    raise Exception(f"file path {full_path} not found")

# R^2 is supposedly 1/pi (?)
R_squared = 1 / 15

'''
print(df) # just checked to make sure data was in the datagram
df_momentum = df.sort_values(by='momentum', ascending=False) # sorted values in the datafram
x = len(df_momentum['momentum']) # finding N value 
x_range = range(1, x+1) # range is x+1 since range doesnt include last number
y = df_momentum['momentum'] # making array of y values
plt.bar(x_range, y) # plotting all y momentum values
#plt.show()

# values for inverse squared plot

y_inverse_square = ((df_momentum['momentum']) ** (-2))
plt.bar(x_range, y_inverse_square) # plotting the inverse square graph
#plt.show()

area = np.trapz(y_inverse_square, x_range) # finding the area under the curve using numpys method
print(f"area: {area}")
N = 66 # setting our N value = 66 
delta_avg2 = (area/N) # calculating delta avg squared
print(delta_avg2)
k1med = np.median(y) # finding median of k1
print(k1med)
avg_dij = pi*k1med*delta_avg2 # calculating average dij with out prediction for R^2=1/pi

print(avg_dij)
'''

# returns the distance between two points in cylindrical coordinates
# assumes rho is 1
def delta_distance(point_i, point_j):

    return (sqrt(
                ((point_i['theta'] - point_j['theta']) ** 2) +
                ((point_i['z'] - point_j['z']) ** 2)
                )
            )

def calculate_d_ij(particle_i, particle_j):
    k_ti = particle_i['momentum']
    k_tj = particle_j['momentum']
    delta_ij = delta_distance(particle_i, particle_j)
    
    d_ij = min((k_ti ** (-2)), (k_tj ** (-2))) * ((delta_ij ** (2)) / (R_squared))

    return d_ij

# Function to sort through groupings of particles according to d_ij
def sort_pairs_by_d_ij(particles_df):
    # Create a list of all possible pairs of particles
    pairs = [(particles_df.iloc[i], particles_df.iloc[j]) for i in range(len(particles_df)) 
             for j in range(i+1, len(particles_df))]

    # Sort pairs by d_ij
    pairs.sort(key=lambda x: calculate_d_ij(x[0], x[1]))

    return pairs
    

def anti_kt(particles_df):

    # initialize groups (each entry assigned an initial group containing only that element)

    particles_df['group'] = 0

    for index, _ in particles_df.iterrows():
        particles_df.at[index, 'group'] = index

    # sort particle groupings by d_ij

    pairs = sort_pairs_by_d_ij(df)

    # iterate through sorted pairs
    for particle_i, particle_j in pairs:
        k_ti = particle_i['momentum']

        d_ij = calculate_d_ij(particle_i, particle_j)

        d_iB = k_ti ** (-2)

        #print(f"d_ij: {d_ij}")
        #print(f"d_iB: {d_iB}")

        # Combine when d_ij < d_iB; stop when d_ij >= d_iB (note: not specified which inequality should be inclusive)

        if (d_ij < d_iB):
            combine_groups(df, particle_i['group'], particle_j['group'])
        else:
            print("STOP")
            break

def combine_groups(df, group_a, group_b):
    group_combined = min(group_a, group_b)

    for index, row in df.iterrows():
        if (row['group'] == group_a or row['group'] == group_b):
            df.at[index, 'group'] = group_combined

# option to test values of R_squared for performance
"""
R_squared_range = np.linspace(0.1, 0.01, 21)
scores = []
for val in R_squared_range:
    R_squared = val
    data = df
    anti_kt(data)
    num_groups = len(np.unique(data['group']))
    # I have injected 5 jets in the data, so I expect groups as the optimal solution
    scores.append(abs((num_groups-5)))

print(scores)"""
# output for the scores: [27, 27, 27, 27, 27, 27, 27, 343, 343, 343, 512, 512, 512, 1331, 1331, 1331, 4096, 4096, 4913, 10648, 21952]
# clearly larger values of R_squared give better scores (lower difference from the truth) but only to some point
#quit()

# used to plot specific runs 

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

for i, txt in enumerate(np.array(df.group)):
    ax.annotate(txt, (z[i], theta[i]))
plt.show()
#combine_groups(df, 2, 3)
