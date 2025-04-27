from setuptools import setup, find_packages

setup(
    name="meta_modal_resonance",
    version="0.1.0",
    packages=find_packages(),
    description="Computational Framework for Meta-Modal Resonance Theory",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Author",
    author_email="author@example.com",
    url="https://github.com/username/meta_modal_resonance",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "networkx>=2.6.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)
