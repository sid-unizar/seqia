from setuptools import setup, find_packages
import glob
import re

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="seqIA",
    version="1.0",
    author="Miguel López-Otal",
    author_email="mlopezotal@unizar.es",
    description="seqIA is a framework for extracting drought impacts and locations from newspaper archives (currently in Spanish).",
    long_description=long_description,
    long_description_content="text/markdown",

    packages=find_packages(),
    package_data={
        'seqia': [
            'models/*',
            'loc_files/*',
        ]
    },
    install_requires=["transformers","accelerate","datasets","tensorflow","numpy","torch","spacy","tqdm","geopy","geopandas","shapely"],
    python_requires=">=3.6"
)
