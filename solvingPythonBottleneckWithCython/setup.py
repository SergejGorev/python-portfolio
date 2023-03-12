from distutils.core import setup, Extension
import numpy as np
from Cython.Distutils import build_ext

engine = Extension("engine",sources=["engine.pyx"],include_dirs=[np.get_include()])

setup( cmdclass={'build_ext': build_ext}, ext_modules=[engine] )