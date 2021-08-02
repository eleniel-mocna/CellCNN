from setuptools import setup

setup(
    name='CellCNN',
    version='0.2',
    description='CNN for sensitive detection in multicell data',
    url='https://github.com/eleniel-mocna/CellCNN',
    install_requires=["tensorflow","matplotlib", "numpy", "pandas"],
    zip_safe=False,
    packages=['CellCNN'],
)