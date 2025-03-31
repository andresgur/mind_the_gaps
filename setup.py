from setuptools import setup, find_packages


setup(
    name="mind_the_gaps",
    author="Andres Gurpide",
    author_email="andres.gurpide@gmail.com",
    description="Package for analysis of irregularly-sampled time series using Gaussian Processes modelling, with focus on period detection in irregularly-sampled time series showing stochastic variability.",
    version="1.0",
    packages=find_packages(),
    scripts=[],
    test_suite='tests',

    keywords=[
        "???"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics"
    ],

    long_description="see README.md",
    install_requires=[
        "stingray>=2.0.0rc1", "numpy>=1.24.4",
        "astropy>=5.3.3", "numexpr>=2.8.7",
        "emcee==3.1.4", "pyfftw>=0.13.0",
        "lmfit==1.0.3", "celerite>=0.4.2", "corner>=2.2.2"
    ],
    url="https://github.com/andresgur/mind_the_gaps"
)