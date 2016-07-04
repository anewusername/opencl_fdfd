#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='opencl_fdfd',
      version='0.1',
      description='Opencl FDFD solver',
      author='Jan Petykiewicz',
      author_email='anewusername@gmail.com',
      url='https://mpxd.net/gogs/jan/opencl_fdfd',
      packages=find_packages(),
      install_requires=[
            'numpy',
            'pyopencl',
            'jinja2',
            'fdfd_tools',
      ],
      extras_require={
      },
      )

