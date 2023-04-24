from setuptools import setup, find_packages

packages = find_packages()

print(packages)
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version = "0.0.1"

setup(
    name="deep_events",
    version=version,
    description="Deep-Events",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LEB-EPFL/deep-events",
    project_urls={
        "Bug Tracker": "https://github.com/LEB-EPFL/deep-events/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    packages=packages,
    package_data={"": ["*.yaml", "database/keys.yaml"], "deep_events": ["*.yaml"]},
    include_package_data=True,
    install_requires=[
        "python_benedict",
        "tifffile",
        "pandas",
        "bson",
        "opencv-python",
        "pymongo",
    ],
    python_requires=">=3.7",
)