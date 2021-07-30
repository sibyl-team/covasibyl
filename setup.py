
from setuptools import setup, find_packages

version = 0.1

setup(
    name="SibylCovasimInterv",
    version=version,
    author="Fabio Mazza and the Sibyl Team",
    description="Interventions for the Covasim model",
    #long_description=long_description,
    #long_description_content_type="text/x-rst",
    #url='http://covasim.org',
    #keywords=["COVID", "COVID-19", "coronavirus", "SARS-CoV-2", "stochastic", "agent-based model", "interventions", "epidemiology"],
    platforms=["OS Independent"],
    #classifiers=CLASSIFIERS,
    packages=find_packages(exclude=["doc","test"]),
    #include_package_data=True,
    install_requires=["numpy", "pandas", "sib", "covasim"],
)