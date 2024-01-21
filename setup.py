from setuptools import find_packages, setup

setup(
    name="adv_gen",
    packages=find_packages(include=["adv_gen"]),
    version="0.1.0",
    description="A package for generating adversarial examples",
    author="Umar Jamil",
    author_email="umarjamil@outlook.com",
    url="https://umarjamil.org",
    install_requires=["torch>=2.0.1"]
)