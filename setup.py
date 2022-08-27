from setuptools import setup

requires = (
    "flask",
    "numpy",
    "opencv-python",
    "tensorflow",
    "keras",
    "tqdm",
    "APScheduler",
)

setup(
    name = 'Image Classifier',
    version='0.1',
    long_description='Cats and dogs image classifier',
    install_requires=requires,
)