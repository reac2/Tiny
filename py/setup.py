from setuptools import setup

setup(
    name='tinymltoolkit',
    version='0.1',
    description='Functions to train and compress neural networks',
    author='Ollie Kemp',
    install_requires=[
        'numpy',
        'tensorflow',
        'optuna',
        'scikit-learn',
    ],
)
