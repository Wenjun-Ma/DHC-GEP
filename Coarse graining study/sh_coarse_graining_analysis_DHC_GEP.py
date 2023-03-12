import os,sys

cell_num_list = [128,64,32,16,8,4]


for cell_num in cell_num_list:
    f=open('Diffusion_equation_DHC-GEP_coarse_graining_analysis.py','r+')
    flist=f.readlines()
    flist[20]= f'cell_num = {cell_num}' + '\n'
    
    f=open('Diffusion_equation_DHC-GEP_coarse_graining_analysis.py','w+')
    f.writelines(flist)
    f.close()

    os.system('python Diffusion_equation_DHC-GEP_coarse_graining_analysis.py')