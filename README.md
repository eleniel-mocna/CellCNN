# CellCNN

CellCNN is a tool for sensitive detection of rare disease-associated cell
subsets via representation learning.

## Instalation

This tool is provided via a docker with an rstudio interface. To run this tool,
install the docker provided here **TODO**.

## Usage

### Simple analysis

The simplest way for analysis with this tool is using the CellCnnAnalysis R6
class interface. It takes a path to an *analysis folder*, with the following structure:

- myFolder/
  - data/
    - first_file.fcs
    - second_file.fcs
    - ...
  - channels.tsv
  - labels.tsv
  - label_description.tsv

#### data folder

In this folder all files used for training should be stored.
All files should be named NAME.fcs, where NAME is used in the labels.tsv file.

#### channels.tsv

This file describes which channels from the fcs files should and should not
be used in the analysis. It must contain the following columns (column names
are in the first row.

- TODO
