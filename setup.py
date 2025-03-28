from setuptools import setup, find_packages
import os

# Read the contents of README.md
with open("readme.md", encoding="utf-8") as f:
    long_description = f.read()

# Read the requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="deepcfr-poker",
    version="0.3.0",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'deepcfr-train=src.training.train:main',
            'deepcfr-play=scripts.play:main',
            'deepcfr-tournament=scripts.visualize_tournament:main',
            'deepcfr-gui=scripts.poker_gui:main'
        ]
    },
    author="Davide Berweger Gaillard",
    author_email="davide@davideb.ch",
    description="Deep CFR Poker AI with Opponent Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players",
    project_urls={
        "Bug Tracker": "https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/issues",
        "Documentation": "https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players#readme",
        "Source Code": "https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.8',
    include_package_data=True,  # Include non-Python files
)