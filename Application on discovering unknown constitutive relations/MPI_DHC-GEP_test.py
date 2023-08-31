import numpy as np
import scipy.io as scio
import os
from numpy import full, linalg as la
from mpi4py import MPI 

comm = MPI.COMM_WORLD
size_comm = comm.Get_size()
rank = comm.Get_rank()

import One_dimensional_shock_parametric_study as DHC_GEP

length_of_head_list = [10,11,12,13,14,15]
genes_in_a_chromosome_list = [1,2]
individuals_in_population_list = [800,1000,1200,1400,1600,1660]
parameter_list = []
for i in range(len(length_of_head_list)):
    for j in range(len(genes_in_a_chromosome_list)):
        for k in range(len(individuals_in_population_list)):
            parameter_list.append([length_of_head_list[i],genes_in_a_chromosome_list[j],individuals_in_population_list[k]])

np.savetxt(f'output/test/{rank}.dat', parameter_list)
with open(f'output/test/{rank}.dat', "a") as f:
    f.write('\n'+ f'{parameter_list[rank][0]}'+ '\n'+ f'{parameter_list[rank][1]}'+ '\n'+ f'{parameter_list[rank][2]}'+ '\n')
