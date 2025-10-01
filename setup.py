from setuptools import setup, find_packages

setup(
    name="latent_ensembles_detector",
    package_dir={"": "src"},
    packages=find_packages(where="src"),           
    version="0.1.0",              
    description="Repo to detect neural ensembles with fast ICA",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Sara Molas Medina",
    url="https://github.com/SaraMolas/latent-ensembles-detector",          # automatically find submodules
    install_requires=[
        # list dependencies here
        "matplotlib",
        "numpy",
        "scipy",
    ],
    python_requires=">=3.8",               # minimum Python version
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # or your license
        "Operating System :: OS Independent",
    ],
)
