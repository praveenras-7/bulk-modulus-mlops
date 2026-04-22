"""
Setup file for bulk-modulus-mlops project.

WHY THIS FILE EXISTS:
- Tells Python "this is an installable package"
- After running 'pip install -e .'
  Python can find src.utils, src.data, etc.
- The -e means "editable" - changes reflect immediately
"""

from setuptools import setup, find_packages

setup(
    name="bulk-modulus-mlops",
    version="0.1.0",
    description="Production MLOps pipeline for bulk modulus prediction",
    author="Praveen M",
    packages=find_packages(),
    python_requires=">=3.9",
)
