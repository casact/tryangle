# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
from setuptools import setup, find_packages

descr = "Tryangle Package - Scientific P&C Loss Reserving"
name = "tryangle"
url = "https://github.com/casact/tryangle"
version = "0.2.1"  # Put this in __init__.py

with open("README.rst") as f:
    long_desc = f.read()

setup(
    name="tryangle",
    maintainer="Caesar Balona",
    maintainer_email="caesar.balona@gmail.com",
    version=version,
    packages=find_packages(where="src", include=["tryangle", "tryangle.*"]),
    package_dir={"": "src"},
    scripts=[],
    url=url,
    license="MPL-2.0",
    description=descr,
    long_description=long_desc,
    long_description_content_type="text/x-rst",
    install_requires=[
        "pandas>=1.0",
        "scikit-learn>=1.0",
        "chainladder>=0.8.12"
    ],
    include_package_data=True,
    package_data={"data": [item for item in os.listdir("src/tryangle/utils/data")]},
    classifiers=[
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
)
