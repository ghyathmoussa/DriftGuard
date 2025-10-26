from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="driftguard",
    version="0.1.0",
    author="Ghyath Moussa",
    author_email="gheathmousa@gmail.com",
    description="ML Model Drift and Performance Monitoring Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ghyathmoussa/DriftGuard",
    packages=find_packages(exclude=["tests*", "examples*", "logs*", "assets*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=0.990",
        ],
    },
    package_data={
        "": ["*.txt", "*.md"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "driftguard-demo=run_demo:main",
        ],
    },
)

