from setuptools import setup, find_packages


setup(
    name="mind_the_gaps",
    author="Andres Gurpide",
    author_email="andres.gurpide@gmail.com",
    description="Functions to manipulate and simulate lightcurves",
    version="1.8",
    packages=find_packages(),
    scripts=[
        "scripts/celerite_script.py",
        "scripts/generate_lcs.py",
        "scripts/fit_lcs.py",
        "scripts/generate_lcs_significance.py",
        "scripts/generate_strategy.py",
        "scripts/plot_ratio_test.py",
    ],
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
        "stingray==2.0.0rc1", "numpy==1.26.4",
        "astropy==5.3.4", "numexpr==2.8.7",
        "emcee==3.1.4", "pyfftw==0.13.0",
        "lmfit==1.0.3", "celerite==0.4.2", "corner==2.2.2"
    ],
    url="https://github.com/andresgur/mind_the_gaps"
)
