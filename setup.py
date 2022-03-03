from setuptools import setup, find_packages
from cedet.__version__ import __version__

requirements = [
    'docopt',
    'h5py',
    'matplotlib',
    'numpy',
    'opencv',
    'pandas',
    'pathlib',
    'scikit-image',
    'scipy',
    'torch',
    'tqdm'
]

setup(
    name='cedet',
    version=__version__,
    description='Center/nucleus detection model selector algorithm.',
    author='James Yu, Vivek Venkatachalam',
    author_email='yu.hyo@northeastern.edu',
    url='https://github.com/venkatachalamlab/cedet',
    entry_points={'console_scripts': ['cedet=cedet.main:main',
                                      'train_cedet=cedet.train:main']},
    keywords=['object detection', 'center detection', 'nucleus detection'],
    # install_requires=requirements,
    packages=find_packages()
)
