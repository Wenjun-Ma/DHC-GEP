# %% import modules
import geppy as gep
from deap import creator, base, tools
import numpy as np
import random
import operator 
import pickle
from fractions import Fraction
import scipy.io as scio
import time
import DHC_GEP as dg

# For reproduction
s = 0
random.seed(s)
np.random.seed(s)

# %% Load data

# Load origin dataset
train_data = scio.loadmat('../data/Diffusion_flow.mat')
rho = train_data['rho']
rho_y = train_data['rho_y']
rho_yy = train_data['rho_yy']
rho_3y = train_data['rho_3y']
Y = train_data['rho_t'] # Target variable is the rho_t.

# Subsample 
train_point_index = list(np.random.randint(len(Y),size = 720))
rho = rho[train_point_index,:]
rho_y = rho_y[train_point_index,:]
rho_yy = rho_yy[train_point_index,:]
rho_3y = rho_3y[train_point_index,:]
Y = Y[train_point_index,:]

diffusion_coef = 2.47e-05 # Diffusion coefficient
# Y = rho_yy * diffusion_coef # If using fully clean data

# %% Creating the primitives set
# define a protected division to avoid dividing by zero
def protected_div(x1, x2):
    if abs(x2) < 1e-10:
        return 1
    return x1 / x2

# Define the operators
pset = gep.PrimitiveSet('Main', input_names=['rho','rho_y','rho_yy','rho_3y'])
pset.add_symbol_terminal('df_c', diffusion_coef)
pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(protected_div, 2)
pset.add_rnc_terminal() # Add random numerical constants (RNC).

# %% Create the individual and population

# Define the indiviudal class, a subclass of gep.Chromosome
creator.create("FitnessMin", base.Fitness, weights=(-1,))  # weights=(-1,)/weights=(1,) means to minimize/maximize the objective (fitness).
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin) 

# Register the individual and population creation operations
h = 3            # head length
n_genes = 1      # number of genes in a chromosome
r = 3            # length of the RNC array
enable_ls = True # whether to apply the linear scaling technique

toolbox = gep.Toolbox()
toolbox.register('rnc_gen', random.randint, a=-10, b=10)   # each RNC is random integer within [-10, 10]
toolbox.register('gene_gen', gep.GeneDc, pset=pset, head_length=h, rnc_gen=toolbox.rnc_gen, rnc_array_length=r)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=None)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register('compile', gep.compile_, pset=pset)# Compile utility: which translates an individual into an executable function (Lambda)

# %% Define the fitness evaluation function

from numba import jit

# Define the loss for individuals that don't apply the linear scaling technique
@jit
def evaluate(individual):
    """Evalute the fitness of an individual: MAE (mean absolute error)"""
    func = toolbox.compile(individual)
    
    # below call the individual as a function over the inputs
    Yp = np.array(list(map(func, rho,rho_y,rho_yy,rho_3y))) 
    
    # return the MRE
    return np.mean(abs((Y - Yp)/Y)),

# Define the loss for individuals that apply the linear scaling technique
@jit
def evaluate_ls(individual):
    """
    First verify whether the individuals satisfy dimensional homogeneity.
    If it is not dimensional homogeneous, we would identify it as an invalid individual and directly assign a significant error to it.
    Otherwise, we would apply linear scaling (ls) to the individual, 
    and then evaluate its loss: MRE (mean relative error)
    """
    func = toolbox.compile(individual)
    Yp = np.array(list(map(func, rho,rho_y,rho_yy,rho_3y)))
    if isinstance(Yp, np.ndarray):
        Q = (np.reshape(Yp, (-1, 1))).astype('float32')
        Q = np.nan_to_num(Q)
        individual.a, residuals, _, _ = np.linalg.lstsq(Q, Y)

        # Define the mean relative error (MRE)
        c = individual.a.reshape(-1,1)
        relative_error = (np.dot(Q,c)-Y)/Y
        if residuals.size > 0:
            return np.mean(abs(relative_error)),

        # Define the relative root mean square error (RRMSE)
        # residuals is the sum of squared errors
        # if residuals.size > 0:
        #     return ((residuals[0])**0.5) / np.linalg.norm(Y, 2),
    
    # regarding the above special cases, the optimal linear scaling w.r.t LSM is just the mean of true target values
    individual.a = 0
    relative_error = (Y-individual.a)/Y
    return np.mean(abs(relative_error)),

if enable_ls:
    toolbox.register('evaluate', evaluate_ls)
else:
    toolbox.register('evaluate', evaluate)

# %% Register genetic operators
toolbox.register('select', tools.selTournament, tournsize=3) # Selection operator
# 1. general operators
toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
toolbox.register('mut_invert', gep.invert, pb=0.1)
toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
toolbox.register('cx_1p', gep.crossover_one_point, pb=0.3)
toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)
# 2. Dc-specific operators
toolbox.register('mut_dc', gep.mutate_uniform_dc, ind_pb=0.05, pb=1)
toolbox.register('mut_invert_dc', gep.invert_dc, pb=0.1)
toolbox.register('mut_transpose_dc', gep.transpose_dc, pb=0.1)
toolbox.register('mut_rnc_array_dc', gep.mutate_rnc_array_dc, rnc_gen=toolbox.rnc_gen, ind_pb='0.5p')
toolbox.pbs['mut_rnc_array_dc'] = 1 

# %% Statistics to be inspected
stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# %% Launch evolution

# Define size of population and number of generations
n_pop = 120              # Number of individuals in a population
n_gen = 100             # Maximum Generation
tol = 1e-3               # Threshold to terminate the evolution
output_type = 'Diffusion_equation_Original-GEP'     # Name of the problem
isRestart = False        # 'True'/'False' for initializing the population with given/random individuals.

# If isRestart is 'True', read the given .pkl file to load the individuals as the first generation population.
# If isRestart is 'False', initialize the first generation population with random individuals.
if isRestart:
    with open("pkl/real-time_Diffusion_equation_Original-GEP.pkl",'rb') as file:
        pop  = pickle.loads(file.read())
else:
    pop = toolbox.population(n=n_pop) 

# Only record the best three individuals ever found in all generations
champs = 3 
hof = tools.HallOfFame(champs)   

# Evolve
start_time = time.time()
pop, log = dg.gep_simple(pop, toolbox, n_generations=n_gen, n_elites=1,
                          stats=stats, hall_of_fame=hof, verbose=True,tolerance = tol,GEP_type = output_type)

# %% Present our Work and Conclusions
# print the No.1 best mathematical expression we found:

elapsed = time.time() - start_time
time_str = '%.2f' % (elapsed)

best_ind = hof[0]
symplified_best = gep.simplify(best_ind)

if enable_ls:
    symplified_best = best_ind.a * symplified_best

key= f'When the evolution is terminated, with CPU running {time_str}s, \nOur No.1 best prediction is:'
with open(f'output/real-time_output_{output_type}.dat', "a") as f:
    f.write('\n'+ key+ str(symplified_best)+ '\n'+f'with error = {toolbox.evaluate(best_ind)[0]}'+'\n')
# %% Present our Work and Conclusions
# print the No.2 best mathematical expression we found:
if str(gep.simplify(hof[1])) != str(gep.simplify(hof[0])):
    best_ind = hof[1]
    symplified_best = gep.simplify(best_ind)

    if enable_ls:
        symplified_best = best_ind.a * symplified_best

    key= f'Our No.2 best prediction is:'
    with open(f'output/real-time_output_{output_type}.dat', "a") as f:
        f.write(key+ str(symplified_best)+ '\n'+f'with error = {toolbox.evaluate(best_ind)[0]}'+'\n')
# %% Present our Work and Conclusions
# print the No.3 best mathematical expression we found:
if str(gep.simplify(hof[2])) != str(gep.simplify(hof[1])) and str(gep.simplify(hof[2])) != str(gep.simplify(hof[0])):
    best_ind = hof[2]
    symplified_best = gep.simplify(best_ind)

    if enable_ls:
        symplified_best = best_ind.a * symplified_best

    key= f'Our No.3 best prediction is:'
    with open(f'output/real-time_output_{output_type}.dat', "a") as f:
        f.write( key+ str(symplified_best)+ '\n'+f'with error = {toolbox.evaluate(best_ind)[0]}'+'\n')