import os,sys

num_molecules_in_cell_list = [400,40]


for num_molecules_in_cell in num_molecules_in_cell_list:
    f=open('Diffusion_equation_DHC-GEP_noise_sensitivity_analysis.py','r+')
    flist=f.readlines()
    flist[20]= f'num_molecules_in_cell = {num_molecules_in_cell}' + '\n'
    
    f=open('Diffusion_equation_DHC-GEP_noise_sensitivity_analysis.py','w+')
    f.writelines(flist)
    f.close()

    os.system('python Diffusion_equation_DHC-GEP_noise_sensitivity_analysis.py')