from setuptools import setup

setup(
    name='provisioning_model',
    url='https://github.com/saviorand/provisioning-model',
    author='Valentin Erokhin',
    author_email='a2svior@gmail.com',
    packages=['model', 'functions', 'notebooks', 'utils'],
    # Needed for dependencies
    install_requires=['dataclasses'],
    # *strongly* suggested for sharing
    version='0.1',
    license='MIT',
    description='An example of a python package from pre-existing code',
    long_description=open('README.txt').read(),
)
