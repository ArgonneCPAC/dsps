from setuptools import setup, find_packages


PACKAGENAME = "dsps"
VERSION = "0.0.dev"


setup(
    name=PACKAGENAME,
    version=VERSION,
    author="Andrew Hearin",
    author_email="ahearin@anl.gov",
    description="Differentiable Stellar Population Synthesis",
    long_description="Differentiable Stellar Population Synthesis",
    install_requires=["numpy", "jax"],
    packages=find_packages(),
    package_data={"dsps": ["tests/testing_data/*.txt", "tests/testing_data/*.dat"]},
    url="https://github.com/ArgonneCPAC/dsps",
)
