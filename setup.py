from numpy.distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

packdata = {}
packdata['pycasso2'] = ['examples/*']

ext_modules = [Extension('pycasso2.resampling_opt',
                         sources=['pycasso2/resampling_opt.pyx']),
               ]

setup(
    name='pycasso2',
    version=0.1,
    packages=['pycasso2'],
    ext_modules=cythonize(ext_modules),
    scripts=[
        'scripts/pycasso_explorer.py',
        'scripts/pycasso_import.py',
        'scripts/pycasso_starlight.py',
        ],
    package_data={'pycasso2': ['data/pycasso.cfg']},
    description="Pycasso2",
    author="Andre Amorim",
    author_email="streetomon@gmail.com",
    url='https://bitbucket.org/streeto/pycasso2',
    platform='Linux',
    license='GPLv3',
)
