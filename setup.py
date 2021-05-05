# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
from setuptools import setup, find_packages

descr = "Tryangle Package - Scientific P&C Loss Reserving"
name = "tryangle"
url = "https://github.com/casact/tryangle"
version = "0.1.0"  # Put this in __init__.py

with open("requirements.txt", "r") as f:
    dependencies = f.read().splitlines()

with open("README.md") as f:
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
    download_url=f"{url}/archive/v{version}.tar.gz",
    license="MPL-2.0",
    description=descr,
    long_description=long_desc,
    install_requires=dependencies,
    include_package_data=True,
    package_data={'data': [item for item in os.listdir("src/tryangle/utils/data")]}
)
