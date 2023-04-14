from setuptools import setup, find_packages


setup(
    name="dsps",
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
