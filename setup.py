from setuptools import setup, find_packages

setup(
    name="bep-epipredict",
    version="1.0.0",
    description="Predicting epigenome editing outcomes using multi-modal deep learning",
    author="BEP-EpiPredict Authors",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0",
        "pyfaidx>=0.7.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "full": ["enformer-pytorch", "shap"],
        "dev": ["pytest", "pytest-cov", "flake8"],
    },
)
