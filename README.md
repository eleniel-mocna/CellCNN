# CellCNN

CellCNN is a tool for sensitive detection of rare disease-associated cell
subsets via representation learning. This method is thoroughly described
[here](https://www.nature.com/articles/ncomms14825) with a followup in [this
paper](https://www.nature.com/articles/s41467-020-18197-y).

## Algorithm Description

This algorithm's main parts are training and evaluation. Firstly the
NN is trained on multiple cell inputs with a fixed size with regards
to the known labels and then all cells in a sample are evaluated based
on the trained filters. These results are then used to classify the samples.

### Training

In the training fase, always N cells (and the labels) are taken from a sample
to form a 'multi cell input'. This is done many times and then the resulting
inputs are fed into a NN, whose first layers consist of Conv1D layers with
the core dimension of 1. Effectively just being dense layers applied
to matrices). To these results is applied RELU. These results are called filters.
These filters are then pooled and classified by a linear classifier or
linear regression for each label.

### Evaluation

After the network is trained, the weights of all layers in front of the filters
are taken and transformed into a NN with a vector as input (This can be done
because of the architecture of Conv1D layers with core dimension of 1). These
then are used to evaluate the filter values for all cells in a sample.
These filter values can then be used either for classification of samples
or any other evaluation.

## Requirements

This tool is developed mainly for a unix server. Docker should provide
sufficient portability, but sometimes it does weird stuff (mainly on Win...)

Also because this tool will run mostly on servers it might not be usable
in low-memory enviroments. The available memory should be at least 2x the
dataset size. I.e: for the 20GB dataset that is provided there should be
at least 40GB of free memory for this tool.
This tool has been tested on a computer with 64 GB of RAM and ran fine
on the provided dataset.

There are no more requirements that I am aware of as of today.

## Instalation

This tool is provided via a docker with an rstudio interface. To run this tool,
use the provided docker.

First build the docker from the provided Dockerfile.

```docker build -t <name_of_the_image> .```

This will take a few tenths of minutes. Then start the container using the following command:

```(bash)
docker run \
  -e PASSWORD=<rocker_password> \
  -p <open_port>:8787 \
  --cpus=<max_CPU_threads> \
  -m=<max_used_memory> -d \
  --name <name_of_the_container> \
  -v <filesystem_mount_directory>:/home/rstudio/data \
  <name_of_the_image>
```

E.g.:

```(bash)
docker run \
  -e PASSWORD=password \
  -p 8787:8787 \
  --cpus=64 \
  -m=200g -d \
  --name CellCNN \
  -v /home:/home/rstudio/data \
  cellcnn
```

Then you can connect to the Rstudio interface inside of this docker using your
browser at: `localhost:<open_port>`.

To login into the rstudio interface use tho following credentials:

- username: `rstudio`
- password: `<rocker_password>`

After the container has started there are a few steps that are needed for the
reticulate R library that have to be taken manually.
To finish the installation, run line-by-line the RInterface/setup.R script.
If some command fails, try restarting the R session then running the step.
If this still doesn't solve the problem, try to restart the R session once
again and then running all the steps in setup.R again.

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

For a detailed description of each file, see below.

When you've prepared this folder, go through the template_example.R script
located in the example_scripts/R folder.

#### data folder

In this folder all files used for training should be stored. The files can
be in one of two supported formats: FCS or CSV.
All files should be named NAME.fcs/NAME.csv, where NAME is used in the
labels.tsv file.

If files are in the CSV format (and you want to use the R interface),
set the `read_csv` in the initializer to `TRUE`.

#### channels.tsv

This tsv file describes which channels from the fcs files should and should not
be used in the analysis. It must contain the following columns (column names
are in the first row.

- name
- use

For example:

| name        | use | foo |
|-------------|-----|-----|
| 1st channel | 0   | bar |
| 2nd channel | 1   | baz |

#### labels.tsv

This tsv file is a matrix of labels for sanmples. Each row corresponds
to one sample and each column corresponds to one label.
First row are label names, first column are sample - file names (that means
that cell (0,0) is empty).

For example:

|             | patient | age |
|-------------|---------|-----|
| first_file  | 0       | 32  |
| second_file | 1       | 64  |

#### label_description.tsv

This tsv file provides information about the type of each used label - that is
whether it describes a binary classification problem, an n-nary classification
problem or a regression problem. It is a table where every row represents
a label and there are 2 mandatory columns:

- name
- type

The (row) name must be exactly the same as the (column) name in the labels.tsv
file.

The type of a label is as follows:

- 0 for regression problems
- 2 for binary classification
- n>2 for n-nary classification
- n<0 for (-n)-nary classification using the earth mover's distance

## Sample Data

Some sample data is available [here](https://cunicz-my.sharepoint.com/:u:/g/personal/73488560_cuni_cz/EfUUA4BZVvZIn2dTGx0q-O0ByZlLn8_tJzhA-85n3viBvQ?e=lBZNs6)
