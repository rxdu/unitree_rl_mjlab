"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "mujoco-warp==3.5.0",
    # mjlab/mujoco-warp only set lower bounds on these, and the latest releases
    # break imports (mujoco 3.14 lacks mjENBL_MULTICCD; warp-lang 1.17 lacks
    # wp.context), so pin to versions matching mujoco-warp 3.5.0 / mjlab 1.2.0.
    "mujoco==3.5.0",
    "warp-lang==1.12.1",
    # Imported by mjlab.terrains but not declared in mjlab's metadata.
    "scipy",
]

# Installation operation
setup(
    name="unitree_rl_mjlab",
    packages=["src"],
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
