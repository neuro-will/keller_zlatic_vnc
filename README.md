# Keller & Zlatic Labs VNC Project

Contains code for analyzing data collected by the Keller & Zlatic labs from the ventral nerve cord of larval Drosphila explants. 

## Dependencies

1) Anaconda
2) janelia_core

## Setup instructions: 

You will need to obtain the source code for the janelia_core porject. Then from the terminal, do the following:

1) Create a conda enviroment: 
	conda create -n keller_zlatic_vnc python=3.7.3 

2) Activate the environment: 
	conda activate keller_zlatic_vnc

3) From the main project directory run:
    python setup.py develop

4) From the janelia_core project directory: 
	Windows/Mac: python setup.py develop

5) Install jupyter notebook:
	conda install jupyter
    

