#!/usr/bin/env python

from setuptools import setup, find_packages
import opencl_fdfd

setup(name='opencl_fdfd',
      version=opencl_fdfd.version,
      description='OpenCL FDFD solver',
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
            'fdfd_tools>=0.3',
      ],
      extras_require={
      },
      )

