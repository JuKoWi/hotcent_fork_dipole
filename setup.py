import re
import sys
from pathlib import Path
from setuptools import Extension, find_packages, setup

# Get the version number:
with open('hotcent/__init__.py') as f:
    version = re.search("__version__ = '(.*)'", f.read()).group(1)

USE_CYTHON = False
if '--use-cython' in sys.argv:
    USE_CYTHON = True
    sys.argv.remove('--use-cython')

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [
    Extension('_hotcent',
              sources=['hotcent/extensions' + ext],
              language='c',
              extra_compile_args=['-O3', '-ffast-math',
                                  '-march=native'],
              ),
]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions, annotate=True)

files = [
    'hotcent-basis',
    'hotcent-concat',
    'hotcent-tables',
]
scripts = [str(Path('tools') / f) for f in files]

install_requires = [
    'ase>=3.21.1',
    'matplotlib',
    'numpy',
    'pytest',
    'pyyaml',
    'scipy',
]

setup(
  ext_modules=extensions,
  install_requires=install_requires,
  license='LICENSE',
  name='hotcent',
  packages=find_packages(),
  scripts=scripts,
  url='https://gitlab.com/mvdb/hotcent',
  version=version,
)
