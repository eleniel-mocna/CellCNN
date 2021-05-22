from setuptools import setup, find_packages

def get_readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='CellCNN',
    version='0.1',
    description='CNN for sensitive detection in multicell data',
    long_description=get_readme(),
    classifiers=[
      'Programming Language :: Python :: 3.7',
    ],
    url='https://github.com/eleniel-mocna/CellCNN',
    include_package_data=True,
    zip_safe=False,
    packages=find_packages(),
)
