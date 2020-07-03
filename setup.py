#!/usr/bin/env python3

from setuptools import setup, find_packages
import opencl_fdfd

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='opencl_fdfd',
      version=opencl_fdfd.version,
      description='OpenCL FDFD solver',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Jan Petykiewicz',
      author_email='anewusername@gmail.com',
      url='https://mpxd.net/code/jan/opencl_fdfd',
      packages=find_packages(),
      package_data={
          'opencl_fdfd': ['kernels/*']
      },
      install_requires=[
            'numpy',
            'pyopencl',
            'jinja2',
            'meanas>=0.5',
      ],
      extras_require={
      },
      )

