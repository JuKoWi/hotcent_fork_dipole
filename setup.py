import sys
from distutils.core import setup
from distutils.extension import Extension

USE_CYTHON = False
if '--use-cython' in sys.argv:
    USE_CYTHON = True
    sys.argv.remove('--use-cython')

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [Extension('_hotcent',
                        sources=['hotcent/extensions' + ext],
                        language='c',
                        extra_compile_args=['-O3', '-ffast-math',
                                            '-march=native'],
                        ),
              ]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions, annotate=True)

setup(
  name='hotcent',
  ext_modules=extensions,
  url='https://gitlab.com/mvdb/hotcent',
  license='LICENSE',
  install_requires=['numpy', 'scipy', 'matplotlib', 'ase', 'pytest'],
)
