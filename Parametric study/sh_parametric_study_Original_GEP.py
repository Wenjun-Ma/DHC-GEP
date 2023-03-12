import os,sys

length_of_head_list = [3,5,7,9,11,13,15]
genes_in_a_chromosome_list = [1,2]
individuals_in_population_list = [200,400,600,800,1000]

for length_of_head in length_of_head_list:
    for genes_in_a_chromosome in genes_in_a_chromosome_list:
        for individuals_in_population in individuals_in_population_list:
            f=open('Diffusion_equation_Original-GEP_parametric_study.py','r+')
            flist=f.readlines()
            flist[63]= f'h = {length_of_head}            # head length' + '\n'
            flist[64]= f'n_genes = {genes_in_a_chromosome}      # number of genes in a chromosome' + '\n'
            flist[65]= f'r = {length_of_head}            # length of the RNC array' + '\n'
            flist[146]= f'n_pop = {individuals_in_population}              # Number of individuals in a population' + '\n'
            if genes_in_a_chromosome == 1:
                flist[71]= f'toolbox.register(\'individual\', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=None)' + '\n'
            else:
                flist[71]= f'toolbox.register(\'individual\', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=operator.add)' + '\n'
            f=open('Diffusion_equation_Original-GEP_parametric_study.py','w+')
            f.writelines(flist)
            f.close()

            os.system('python Diffusion_equation_Original-GEP_parametric_study.py')