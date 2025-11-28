from setuptools import setup, find_packages

setup(
    name="neurovisus",
    version="1.0.0",
    author="Your Name",
    description="Multimodal Pain Assessment System",
    packages=find_packages(),
    install_requires=[
        "torch", "torchvision", "numpy", "opencv-python", "tqdm", "pyyaml"
    ],
)