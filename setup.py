from setuptools import setup, find_packages


PACKAGENAME = "dsps"
VERSION = "0.1.0"


setup(
    name=PACKAGENAME,
    version=VERSION,
    author="Andrew Hearin",
    author_email="ahearin@anl.gov",
    description="Differentiable Stellar Population Synthesis",
    long_description="Differentiable Stellar Population Synthesis",
    install_requires=["numpy", "jax", "diffmah"],
    packages=find_packages(),
    package_data={
        "dsps": ["data/*.npy", "tests/testing_data/*.txt", "tests/testing_data/*.dat"]
    },
    url="https://github.com/ArgonneCPAC/dsps",
)
