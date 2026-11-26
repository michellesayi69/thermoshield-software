"""ThermoShield packaging configuration.

Uses classic setuptools setup.py rather than PEP 621 pyproject.toml so
the package installs cleanly on pip 21.2.4 (the system default on the
target macOS machine). Migration to pyproject.toml is straightforward
once pip is upgraded; it is not a priority for the current scope.
"""
from setuptools import setup, find_packages

setup(
    name="thermoshield",
    version="0.1.0",
    description="Threshold-based thermal anomaly detection simulation",
    author="Michelle Sayi",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
        "matplotlib>=3.5",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
)
