from setuptools import setup
import os
import atexit


def cleanup():
    # Kill the current process
    os.kill(os.getpid(), 9)


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
    cmdclass={
        'install': atexit.register(cleanup)
    }
)
