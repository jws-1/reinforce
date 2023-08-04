from setuptools import setup, find_packages

# Read the dependencies from requirements.txt
with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(
    name='reinforce',
    version='0.0.1',
    description='A Python package for creating and benchmarking RL agents.',
    author='Jared Swift',
    author_email='j.w.swift@outlook.com',
    url='https://github.com/jws-1/reinforce',
    packages=find_packages(),
    install_requires=required_packages,
)

