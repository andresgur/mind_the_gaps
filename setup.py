from setuptools import setup


setup(name="mind_the_gaps", author="Andres Gurpide", author_email="andres.gurpide@gmail.com",
     description="Functions to manipulate and simulate lightcurves", version="1.8",
      packages=["mind_the_gaps"], long_description="see README.md", install_requires=["stingray==1.1.2", "numpy==1.24.4",
      "astropy==5.3.3", "numexpr==2.7.3", "emcee==3.1.4", "pyfftw==0.13.0", "lmfit==1.0.3", "celerite==0.4.2", "corner==2.2.2"],
      url="https://github.com/andresgur/mind_the_gaps")
