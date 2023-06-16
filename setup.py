import io
import os
from setuptools import find_packages, setup

MAJOR = "0"
MINOR = "0.1"

if os.path.exists("../major.version"):
    with open("../major.version", "rt") as bf:
        MAJOR = str(bf.read()).strip()

if os.path.exists("../minor.version"):
    with open("../minor.version", "rt") as bf:
        MINOR = str(bf.read()).strip()

VERSION = "{}.{}".format(MAJOR, MINOR)

with io.open("README.md", "r", encoding="utf-8") as f:
    README = f.read()

setup(
    name="omop-learn",
    version=VERSION,
    description="Machine learning on healthcare data stored in OHDSI's OMOP CDM.",
    long_description=README,
    long_description_content_type="text/x-rst",
    namespace_packages=[],
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=["test*"]),
    include_package_data=True,
    python_requires=">=3.5,<4",
)
