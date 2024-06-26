from setuptools import setup, find_packages

setup(
    name="embeddings",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # Add other setup arguments as needed
)
