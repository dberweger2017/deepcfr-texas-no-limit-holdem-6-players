from setuptools import setup, find_packages

setup(
    name="deepcfr-poker",
    version="0.2.0",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'numpy',
        'matplotlib',
        'tensorboard',
        'argparse',
        'tqdm',
        'pokers',
        'python-dotenv',
        'requests',
        'seaborn',
        'pandas'
    ],
    entry_points={
        'console_scripts': [
            'deepcfr-train=src.training.train:main',
            'deepcfr-play=scripts.play:main',
            'deepcfr-tournament=scripts.visualize_tournament:main'
        ]
    },
    author="Davide Berweger Gaillard",
    description="Deep CFR Poker AI with Opponent Modeling",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dberweger2017/deepcfr-poker",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)