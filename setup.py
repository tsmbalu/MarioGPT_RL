import io
import os
import re
from os import path

from setuptools import find_packages
from setuptools import setup


def parse_requirements(filename):
    """ Load requirements from a pip requirements file """
    with open(filename, 'r') as f:
        return f.read().splitlines()

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name="mario-gpt",
    version="0.1.3",
    url="https://github.com/shyamsn97/mario-gpt",
    license='MIT',

    author="Shyam Sudhakaran",
    author_email="shyamsnair@protonmail.com",

    description="Generating Mario Levels with GPT2. Code for the paper: 'MarioGPT: Open-Ended Text2Level Generation through Large Language Models', https://arxiv.org/abs/2302.05981",

    long_description=long_description,
    long_description_content_type="text/markdown",

    include_package_data = True,

    packages=find_packages(exclude=('tests',)),

    install_requires=parse_requirements('requirements.txt'),

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
