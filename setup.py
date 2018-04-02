from numpy.distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

ext_modules = [Extension('pycasso2.resampling.hist_resample_opt',
                         sources=['pycasso2/resampling/hist_resample_opt.pyx']),
               Extension('pycasso2.resampling.gauss_smooth',
                         sources=['pycasso2/resampling/gauss_smooth.pyx']),
               Extension('pycasso2.external.photutils.circular_overlap',
                         sources=['pycasso2/external/photutils/circular_overlap.pyx']),
               Extension('pycasso2.external.photutils.elliptical_exact',
                         sources=['pycasso2/external/photutils/elliptical_exact.pyx']),
               Extension('pycasso2.external.photutils.downsample',
                         sources=['pycasso2/external/photutils/downsample.pyx']),
               ]

setup(
    name='pycasso2',
    version='2.0.0',
    packages=['pycasso2',
              'pycasso2.dobby',
              'pycasso2.dobby.models',
              'pycasso2.external',
              'pycasso2.external.pylygon',
              'pycasso2.external.photutils',
              'pycasso2.importer',
              'pycasso2.legacy',
              'pycasso2.starlight',
              ],
    ext_modules=cythonize(ext_modules),
    scripts=[
        'scripts/pycasso_explorer.py',
        'scripts/pycasso_import.py',
        'scripts/pycasso_starlight.py',
        'scripts/pycasso_convert.py',
        ],
    package_data={'pycasso2': ['data/pycasso.cfg.template'],
                  'pycasso2': ['starlight/gridfile.template']},
    description="Pycasso2",
    author="Andre Amorim",
    author_email="streetomon@gmail.com",
    url='https://bitbucket.org/streeto/pycasso2',
    platform='Linux',
    license='GPLv3',
)
