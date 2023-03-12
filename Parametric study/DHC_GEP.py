# coding=utf-8

'''
    .. moduleauthor:: Wenjun Ma
This :mod:`DHC_GEP` module provides the addtional algrithms of DHC-GEP to Original-GEP, 
including the dimensional verification operation, real-time output, termination criterion (with given threshold) 
and real-time saving populations for restarting.
'''

from fractions import Fraction
import os

def dimensional_verification(individual, dict_of_dimension, target_dimension):

    def my_add(a,b):

        if isinstance(a,bool) or isinstance(b,bool):
            return False
        if isinstance(a,int):
            a = Fraction(1)
        if isinstance(b,int):
            b = Fraction(1)

        if a == b:
            return a
        else:
            return False

    def my_sub(a,b):
        
        if isinstance(a,bool) or isinstance(b,bool):
            return False
        if isinstance(a,int):
            a = Fraction(1)
        if isinstance(b,int):
            b = Fraction(1)

        if a == b:
            return a
        else:
            return False

    def my_mul(a,b):
        
        if isinstance(a,bool) or isinstance(b,bool):
            return False
        if isinstance(a,int):
            a = Fraction(1)
        if isinstance(b,int):
            b = Fraction(1)

        return a * b

    def my_truediv(a,b):
        
        if isinstance(a,bool) or isinstance(b,bool) or b == 0:
            return False
        if isinstance(a,int):
            a = Fraction(1)
        if isinstance(b,int):
            b = Fraction(1)

        return a / b
    
    individual_expr = individual.__str__().replace('\t','').replace('\n','').replace('add','my_add').replace('sub','my_sub').replace('mul','my_mul').replace('truediv','my_truediv')
    
    create_var = locals()
    create_var.update(dict_of_dimension)
    
    dimension_of_DDEq = eval(individual_expr)
    if dimension_of_DDEq == target_dimension:
        return True
    else:
        return False

"""
The original author of the following functions is Shuhua Gao, which can be here https://github.com/ShuhuaGao/geppy/blob/master/geppy/algorithms/basic.py.
Wenjun Ma modified the function 'gep_simple', adding the following features:
    * Output the real-time resultant mathematical expressions to a .dat file.
    * Write the populations to a .pkl file every 20 generations for ease of subsequent restarting if necessary.
    * Terminate evolution with given threshold. When the error of the best individual is smaller than the threshold, the evolution is terminated. 

Following module provides fundamental boilerplate GEP algorithm implementations. After registering proper
operations into a :class:`deap.base.Toolbox` object, the GEP evolution can be simply launched using the present
algorithms. Of course, for complicated problems, you may want to define your own algorithms, and the implementation here
can be used as a reference.
"""
import deap
import random
import warnings
import numpy as np
import pickle
import datetime
import geppy as gep
import time
# from builtins import str

def _validate_basic_toolbox(tb):
    """
    Validate the operators in the toolbox *tb* according to our conventions.
    """
    assert hasattr(tb, 'select'), "The toolbox must have a 'select' operator."
    # whether the ops in .pbs are all registered
    for op in tb.pbs:
        assert op.startswith('mut') or op.startswith('cx'), "Operators must start with 'mut' or 'cx' except selection."
        assert hasattr(tb, op), "Probability for a operator called '{}' is specified, but this operator is not " \
                                "registered in the toolbox.".format(op)
    # whether all the mut_ and cx_ operators have their probabilities assigned in .pbs
    for op in [attr for attr in dir(tb) if attr.startswith('mut') or attr.startswith('cx')]:
        if op not in tb.pbs:
            warnings.warn('{0} is registered, but its probability is NOT assigned in Toolbox.pbs. '
                          'By default, the probability is ZERO and the operator {0} will NOT be applied.'.format(op),
                          category=UserWarning)


def _apply_modification(population, operator, pb):
    """
    Apply the modification given by *operator* to each individual in *population* with probability *pb* in place.
    """
    for i in range(len(population)):
        if random.random() < pb:
            population[i], = operator(population[i])
            del population[i].fitness.values
    return population


def _apply_crossover(population, operator, pb):
    """
    Mate the *population* in place using *operator* with probability *pb*.
    """
    for i in range(1, len(population), 2):
        if random.random() < pb:
            population[i - 1], population[i] = operator(population[i - 1], population[i])
            del population[i - 1].fitness.values
            del population[i].fitness.values
    return population


def gep_simple(population, toolbox, n_generations=100, n_elites=1,
               stats=None, hall_of_fame=None, verbose=__debug__,tolerance = 1e-10,GEP_type = '', length_of_head = 0, genes_in_a_chromosome = 0, individuals_in_population = 0):
    """
    This algorithm performs the simplest and standard gene expression programming.
    The flowchart of this algorithm can be found
    `here <https://www.gepsoft.com/gxpt4kb/Chapter06/Section1/SS1.htm>`_.
    Refer to Chapter 3 of [FC2006]_ to learn more about this basic algorithm.

    .. note::
        The algorithm framework also supports the GEP-RNC algorithm, which evolves genes with an additional Dc domain for
        random numerical constant manipulation. To adopt :func:`gep_simple` for GEP-RNC evolution, use the
        :class:`~geppy.core.entity.GeneDc` objects as the genes and register Dc-specific operators.
        A detailed example of GEP-RNC can be found at `numerical expression inference with GEP-RNC
        <https://github.com/ShuhuaGao/geppy/blob/master/examples/sr/numerical_expression_inference-RNC.ipynb>`_.
        Users can refer to Chapter 5 of [FC2006]_ to get familiar with the GEP-RNC theory.

    :param population: a list of individuals
    :param toolbox: :class:`~geppy.tools.toolbox.Toolbox`, a container of operators. Regarding the conventions of
        operator design and registration, please refer to :ref:`convention`.
    :param n_generations: max number of generations to be evolved
    :param n_elites: number of elites to be cloned to next generation
    :param stats: a :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param hall_of_fame: a :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: whether or not to print the statistics.
    :returns: The final population
    :returns: A :class:`~deap.tools.Logbook` recording the statistics of the
              evolution process

    .. note:
        To implement the GEP-RNC algorithm for numerical constant evolution, the :class:`geppy.core.entity.GeneDc` genes
        should be used. Specific operators are used to evolve the Dc domain of :class:`~geppy.core.entity.GeneDc` genes
        including Dc-specific mutation/inversion/transposition and direct mutation of the RNC array associated with
        each gene. These operators should be registered into the *toolbox*.
    """
    _validate_basic_toolbox(toolbox)
    logbook = deap.tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    start_time = time.time()

    is_exists = os.path.exists('pkl')
    if not is_exists:
        os.mkdir('pkl')
    
    is_exists = os.path.exists('output')
    if not is_exists:
        os.mkdir('output')
    
    simplified_best_list = []
    for gen in range(n_generations + 1):
        # evaluate: only evaluate the invalid ones, i.e., no need to reevaluate the unchanged ones
        invalid_individuals = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit

        # record statistics and log
        if hall_of_fame is not None:
            hall_of_fame.update(population)
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_individuals), **record)
        if verbose:
            print(logbook.stream)

        if gen == n_generations:
            break

        # selection with elitism
        elites = deap.tools.selBest(population, k=n_elites)
        offspring = toolbox.select(population, len(population) - n_elites)

        # output the real-time result
        # Every 20 generations, the current optimal individual is checked, and if a new optimal individual appears, it is output to file.
        if gen > 0 and gen % 20 == 0:
            elites_IR = elites[0]
            simplified_best = gep.simplify(elites_IR)
            if str(simplified_best) not in simplified_best_list:
                simplified_best_list.append(str(simplified_best))
                elapsed = time.time() - start_time
                time_str = '%.2f' % (elapsed)   
                if elites_IR.a != 1e18:
                    simplified_best = elites_IR.a * simplified_best      
                    key= f'With length_of_head = {length_of_head}, genes_in_a_chromosome = {genes_in_a_chromosome}, individuals_in_population = {individuals_in_population}\nIn generation {gen}, with CPU running {time_str}s, \nOur No.1 best prediction is:'
                    with open(f'output/{GEP_type}_equation.dat', "a") as f:
                        f.write('\n'+ key+ str(simplified_best)+ '\n'+f'with loss = {elites_IR.fitness.values[0]}'+'\n')
                else:
                    key= f'With length_of_head = {length_of_head}, genes_in_a_chromosome = {genes_in_a_chromosome}, individuals_in_population = {individuals_in_population}\nIn generation {gen}, with CPU running {time_str}s, \nOur No.1 best prediction 1 is:'
                    with open(f'output/{GEP_type}_equation.dat', "a") as f:
                        f.write('\n'+ key+ str(simplified_best)+ '\n'+f'which is invalid!'+'\n' )

        # Termination criterion of error tolerance
        if gen > 0 and gen % 100 == 0:
            error_min = elites[0].fitness.values[0]
            if error_min<tolerance:
                time_now = str(datetime.datetime.now())
                pklFileName = time_now[:16].replace(':', '_').replace(' ', '_')
                output_hal = open(f'pkl/{GEP_type}.pkl', 'wb')
                str_class = pickle.dumps(population)
                output_hal.write(str_class)
                output_hal.close()
                break

        # # Termination criterion of error reducing velocity
        # # Compute the best individual every 500 generations, if the best 
        # # one is the same with that of the 500 generations before, 
        # # terminate the evolution immediately.
        # if gen % 30 == 0:
        #     elites_IR = elites[0]
        #     middle_simplified_best = gep.simplify(elites_IR)
        #     if gen == 0:
        #         realtime_middle_simplified_best = middle_simplified_best
        #     else:
        #         if realtime_middle_simplified_best == middle_simplified_best:
        #             output_hal = open(f'pkl/{GEP_type}.pkl', 'wb')
        #             str_class = pickle.dumps(population)
        #             output_hal.write(str_class)
        #             output_hal.close()
        #             break
        #         else:
        #             realtime_middle_simplified_best = middle_simplified_best

        # replication
        offspring = [toolbox.clone(ind) for ind in offspring]

        # mutation
        for op in toolbox.pbs:
            if op.startswith('mut'):
                offspring = _apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op])

        # crossover
        for op in toolbox.pbs:
            if op.startswith('cx'):
                offspring = _apply_crossover(offspring, getattr(toolbox, op), toolbox.pbs[op])

        # replace the current population with the offsprings
        population = elites + offspring
    
    time_now = str(datetime.datetime.now())
    pklFileName = time_now[:16].replace(':', '_').replace(' ', '_')
    output_hal = open(f'pkl/{GEP_type}.pkl', 'wb')
    str_class = pickle.dumps(population)
    output_hal.write(str_class)
    output_hal.close()

    return population, logbook


__all__ = ['gep_simple']


