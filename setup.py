from setuptools import setup



with open("requirements.txt", 'r') as dependencies:
    requirements = [pkg.strip() for pkg in dependencies]

setup(
    name='tinymltoolkit',
    version='0.1',
    description='Functions to train and compress neural networks',
    author='Ollie Kemp',
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires='>=3.10',
)
