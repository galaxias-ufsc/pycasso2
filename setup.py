from numpy.distutils.core import setup

packdata = {}
packdata['pycasso2'] = ['examples/*']

setup(
    name='pycasso2',
    version=0.1,
    packages=['pycasso2'],
    scripts=[
        'scripts/pycasso_explorer.py',
        'scripts/pycasso_import.py',
        'scripts/pycasso_starlight.py',
        'scripts/pycasso_segment.py',
        ],
    description="Pycasso2",
    author="André Amorin",
    author_email="streetomon@gmail.com",
    url='https://bitbucket.org/streeto/pycasso2',
    platform='Linux',
    license='GPLv3',
)
