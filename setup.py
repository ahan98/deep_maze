from setuptools import setup

setup(
    name="cair_maze",
    version="0.0.1",
    install_requires=["gymnasium==0.28.1", "pygame==2.1.2"],
    packages=["cair_maze", "cair_maze.envs", "dream"]
)
