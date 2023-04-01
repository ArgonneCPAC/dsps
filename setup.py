import os
from setuptools import setup, find_packages


__version__ = None
pth = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dsps", "_version.py")
with open(pth, "r") as fp:
    exec(fp.read())


setup(
    name="dsps",
    version=__version__,
    author="Andrew Hearin",
    author_email="ahearin@anl.gov",
    description="Differentiable Stellar Population Synthesis",
    long_description="Differentiable Stellar Population Synthesis",
    install_requires=["numpy", "jax"],
    packages=find_packages(),
    package_data={
        "dsps": [
            "data/*.npy",
            "tests/testing_data/*.txt",
            "tests/testing_data/*.dat",
            "*/tests/testing_data/*.txt",
            "*/tests/testing_data/*.dat",
        ]
    },
    url="https://github.com/ArgonneCPAC/dsps",
)
