from setuptools import setup, find_packages

setup(
    name="delta-phase",
    version="1.0.0",
    description="High-Expressivity O(N) Complex Phase Matrix Delta-Rule Memory & Lerp Spectral LLM",
    author="M. Carbonell",
    url="https://github.com/mcarbonell/delta-phase",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
