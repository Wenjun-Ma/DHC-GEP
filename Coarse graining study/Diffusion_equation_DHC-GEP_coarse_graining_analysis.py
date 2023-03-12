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
cell_num = 4
train_data = scio.loadmat(f'../data/Coarse_graining/Diffusion_flow_cell_num_{cell_num}.mat')
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

diffusion_coef = 1.399e-05 # Diffusion coefficient
miu = 2.079e-5 # Viscosity coefficient
# Y = rho_yy * diffusion_coef # If using fully clean data

# %% Assign number tags

# Assign prime number tags to base dimensions
L,M,T,I,Theta,N,J = 2,3,5,7,11,13,17

# Derive the tags for dirived physical quantities according to their dimensions
# Note that the tags are always in the form of fractions, instead of floats, which avoids introducing any truncation errors. 
# Therefore, we use 'Fraction' function here.
dict_of_dimension = {'rho':Fraction(M,((L)**(3))),
                     'rho_y':Fraction(M,((L)**(4))),
                     'rho_yy':Fraction(M,((L)**(5))),
                     'rho_3y':Fraction(M,((L)**(6))),
                     'df_c':Fraction((L**2),T),
                     'Miu':Fraction(M,L*T)} 

# Assign number tags to taget variable
target_dimension = Fraction(M,T*((L)**(3)))

# %% Creating the primitives set
# define a protected division to avoid dividing by zero
def protected_div(x1, x2):
    if abs(x2) < 1e-10:
        return 1
    return x1 / x2

# Define the operators
pset = gep.PrimitiveSet('Main', input_names=['rho','rho_y','rho_yy','rho_3y'])
pset.add_symbol_terminal('df_c', diffusion_coef)
pset.add_symbol_terminal('Miu', miu)
pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(operator.truediv, 2)
pset.add_rnc_terminal() # Add random numerical constants (RNC).
np.seterr(divide='raise')

# %% Create the individual and population

# Define the indiviudal class, a subclass of gep.Chromosome
creator.create("FitnessMin", base.Fitness, weights=(-1,))  # weights=(-1,)/weights=(1,) means to minimize/maximize the objective (fitness).
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin) 

# Register the individual and population creation operations
h = 15            # head length
n_genes = 2      # number of genes in a chromosome
r = 15            # length of the RNC array
enable_ls = True # whether to apply the linear scaling technique

toolbox = gep.Toolbox()
toolbox.register('rnc_gen', random.randint, a=-10, b=10)   # each RNC is random integer within [-10, 10]
toolbox.register('gene_gen', gep.GeneDc, pset=pset, head_length=h, rnc_gen=toolbox.rnc_gen, rnc_array_length=r)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=operator.add)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register('compile', gep.compile_, pset=pset)# Compile utility: which translates an individual into an executable function (Lambda)

# %% Define the loss function

# Register the dimensional verification operation
toolbox.register('dimensional_verification', dg.dimensional_verification)

# Define the loss for individuals that don't apply the linear scaling technique
def evaluate(individual):
    """Evalute the fitness of an individual: MAE (mean absolute error)"""
    func = toolbox.compile(individual)
    
    # below call the individual as a function over the inputs
    Yp = np.array(list(map(func, rho,rho_y,rho_yy,rho_3y))) 
    
    # return the MRE
    return np.mean(abs((Y - Yp)/Y)),

# Define the loss for individuals that apply the linear scaling technique
def evaluate_ls(individual):
    """
    First verify whether the individuals satisfy dimensional homogeneity.
    If it is not dimensional homogeneous, we would identify it as an invalid individual and directly assign a significant error to it.
    Otherwise, we would apply linear scaling (ls) to the individual, 
    and then evaluate its loss: MRE (mean relative error)
    """
    validity = toolbox.dimensional_verification(individual, dict_of_dimension, target_dimension)
    if not validity:
        individual.a = 1e18
        return 1000,
    else:
        func = toolbox.compile(individual)
        try:
            Yp = np.array(list(map(func, rho,rho_y,rho_yy,rho_3y)))
        except FloatingPointError:
            individual.a = 1e18
            return 1000,
        except ZeroDivisionError:
            individual.a = 1e18
            return 1000,
        if isinstance(Yp, np.ndarray):
            Q = (np.reshape(Yp, (-1, 1))).astype('float32')
            Q = np.nan_to_num(Q)
            individual.a, residuals, _, _ = np.linalg.lstsq(Q, Y)

            # Define the mean relative error (MRE)
            c = individual.a.reshape(-1,1)
            relative_error = (np.dot(Q,c)-Y)/Y
            if residuals.size > 0:
                return np.mean(abs(relative_error)),

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
n_pop = 1660              # Number of individuals in a population
n_gen = 100             # Maximum Generation
tol = 1e-3               # Threshold to terminate the evolution
output_type = f'Diffusion_equation_DHC-GEP_coarse_graining_analysis_{cell_num}'     # Name of the problem
isRestart = False        # 'True'/'False' for initializing the population with given/random individuals.

# If isRestart is 'True', read the given .pkl file to load the individuals as the first generation population.
# If isRestart is 'False', initialize the first generation population with random individuals.
if isRestart:
    with open(f"pkl/real-time_Diffusion_eauqtion_DHC-GEP_coarse_graining_analysis_{cell_num}.pkl",'rb') as file:
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