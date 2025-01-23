from setuptools import setup, find_packages

setup(
    name="hangman",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pytorch-lightning",
        "wandb",
        "pandas",
        "matplotlib",
        "seaborn",
        "numpy",
        "tqdm",
    ],
) 