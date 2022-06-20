from setuptools import find_packages
from setuptools import setup


setup(name = 'modularity_aware_gae',
      description = 'Modularity-Aware Graph Autoencoders for Joint Community Detection and Link Prediction',
      author = 'G. Salha-Galvan et al. (Neural Networks journal, Elsevier, 2022)',
      install_requires = ['networkx==2.4',
                          'numpy==1.19.5',
                          'python-louvain==0.14',
                          'scikit-learn==0.22.1',
                          'scipy==1.4.1',
                          'tensorflow==1.15'],
      package_data = {'modularity_aware_gae': ['README.md']},
      packages = find_packages())