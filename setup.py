from setuptools import setup

version = '1.0.0'

setup(
    name='ProcessPruner',
    version=version,
    description='Process Pruner is an automated preprocessing tool for filtering and '
                'highlighting of insights in an event log',
    install_requires=[
        'pandas', 'tqdm', 'numpy'
    ],
    scripts=[],
)
