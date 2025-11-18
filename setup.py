import setuptools

with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="ampharos",
    version="0.0.1",
    author="Mathijs Vanrespaille",
    author_email="mathijs.vanrespaille@kuleuven.be",
    packages=["ampharos"],
    include=["ampharos/__init__.py", "ampharos/instability_strips/", "ampharos/io_functions.py", "ampharos/support_functions.py", "ampharos/visualisation.py"],
    exclude=["ampharos/identify_storm_output.py", "ampharos/main_functions.py"],
#    description="Asteroseismic forward modelling of beta Cephei pulsators",
    description="Asteroseismic Modelling of Hot Pulsators Accounting for Rotation using Oscillation code StORM",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mathijs-Vanrespaille/AMPHAROS.git",
    license='GNU GENERAL PUBLIC LICENSE Version 3',
    python_requires='>=3.9',
    install_requires=["matplotlib >= 3.5", "h5py >= 3.12", "numpy >= 1.26", "scipy >= 1.13"],
    extras_require={"first_order_rotation_treatment":"pygyre"}
)
