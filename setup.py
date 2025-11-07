import setuptools

with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="",
    version="0.0.1",
    author="Mathijs Vanrespaille",
    author_email="mathijs.vanrespaille@kuleuven.be",
    packages=["ampharos"],
    description="Asteroseismic forward modelling of beta Cephei pulsators",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mathijs-Vanrespaille/AMPHAROS.git",
    license='GNU GENERAL PUBLIC LICENSE Version 3',
    python_requires='>=3.9',
    install_requires=["matplotlib", "h5py", "numpy", "scipy"],
    extras_require={"first_order_rotation_treatment":"pygyre"}
)
