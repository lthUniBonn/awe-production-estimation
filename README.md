# Production estimation from input wind profile using quasi steady model for AWE soft wing


## Installing and running the code
### Creating the conda environment
To create the conda environment run
'''conda create -p [path/envName] --file requirements.txt'''

#### Install pyoptsparse into the environment
*Full:*
Activate the conda environment.
'''pip install git+https://github.com/mdolab/pyoptsparse@v2.5.1'''
*Light:*
This installs less modules, and does not require swig.
Clone the pyoptsparse git repository. 
Select version via 
'''git checkout v2.5.1'''
Comment out the lines concerning NOMAD and NSGA2 in the 'pyoptsparse/pyoptsparse/setup.py' and 
'pyoptsparse/pyoptsparse/__init__.py'.
Activate the conda environment, navigate to the pyoptsparse top level folder, run
'''pip install .'''
The pyoptsparse package should now be available in the environment. 
