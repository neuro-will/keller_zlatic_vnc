# Keller & Zlatic Labs VNC Project

Contains code for analyzing data collected by the Keller & Zlatic labs from the ventral nerve cord of larval Drosphila explants. 

## Dependencies

1) Anaconda
2) janelia_core

## Setup instructions: 

We assume you have conda installed.  You can obtain it [here](https://www.anaconda.com/products/distribution).

Once conda is installed, follow these instructions to setup the project code and all required dependencies:

1) Create a conda enviroment: 
	conda create -n keller_zlatic_vnc python=3.8

2) Activate the environment: 
	conda activate keller_zlatic_vnc

3) Install the janelia_core package into the conda environment you just created.  It can be obtained [here](https://github.com/wbishopJanelia/janelia_core), and instructions to install it can be found [here](https://wbishopjanelia.github.io/janelia_core/).


4) From the main project directory for keller_zlatic_vnc run:
    python setup.py develop


5) Install jupyter notebook:
	conda install jupyter
    

