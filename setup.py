import setuptools

with open("tutorial.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="morning",
    version="0.0.1",
    author="Mathijs Vanrespaille",
    author_email="mathijs.vanrespaille@kuleuven.be",
    packages=["morning"],
    description="Asteroseismic forward modelling of beta Cephei pulsators",
    long_description=description,
    long_description_content_type="text/markdown",
    url="",
    license='GNU GENERAL PUBLIC LICENSE Version 3',
    python_requires='>=3.9',
    install_requires=[]
)
