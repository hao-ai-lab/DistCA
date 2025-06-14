from setuptools import setup, find_packages

setup(
    name="d2",
    version="0.1",
    packages=find_packages(),  # Automatically find all packages with __init__.py
    install_requires=[
        # List your dependencies or read from requirements.txt
    ],
    python_requires='>=3.8',
)