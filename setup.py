# my-lib/setup.py

from setuptools import setup, find_packages

setup(
    name="mylib",
    version="0.1",
    packages=find_packages(),  # Automatically finds 'mylib'
    install_requires=[],       # Add dependencies if needed
)
