import setuptools

with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="ampharos",
    version="0.1.1",
    author="Mathijs Vanrespaille",
    author_email="mathijs.vanrespaille@kuleuven.be",
    packages=["ampharos"],
    include=["ampharos/__init__.py", "ampharos/instability_strips/", "ampharos/io_functions.py", "ampharos/support_functions.py", "ampharos/visualisation.py"],
    exclude=["ampharos/identify_storm_output.py", "ampharos/main_functions.py"],
    description="Asteroseismic Modelling of Hot Pulsators Accounting for Rotation using Oscillation code StORM",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mathijs-Vanrespaille/AMPHAROS.git",
    license='GNU GENERAL PUBLIC LICENSE Version 3',
    python_requires='>=3',
    install_requires=["matplotlib", "h5py", "numpy", "pandas", "scipy"],
    extras_require={"first_order_rotation_treatment":"pygyre"}
)
