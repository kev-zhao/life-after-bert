from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().strip().splitlines()

setup(
    name="life_after_bert",
    version="0.1",
    url="https://github.com/kev-zhao/life-after-bert",
    install_requires=requirements,
)
