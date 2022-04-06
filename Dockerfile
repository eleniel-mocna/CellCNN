FROM rocker/rstudio:4.0.5
LABEL NAME="cellcnn_eleniel" Version="1.1"
LABEL author="Samuel Soukup"
LABEL contact="soukup.sam(at)gmail.com"

# CellCnn folder should be mounted via -v

RUN mkdir /DockerBuildInfo
COPY Dockerfile /DockerBuildInfo

COPY README.md /README.md
COPY CellCNN /backend/CellCNN
COPY setup.py /backend/setup.py
COPY requirements.txt /backend/requirements.txt
COPY RInterface /RInterface

RUN apt-get update && \
    apt-get install -y zlib1g-dev && \
    apt-get clean

# RUN Rscript /RInterface/install.R

RUN r -e "install.packages('BiocManager')"
RUN r -e "BiocManager::install('flowCore')"
RUN r -e "install.packages('reticulate')"
RUN r -e "install.packages('glue')"
RUN r -e "install.packages('R6')"
RUN r -e "install.packages('stringr')"
RUN r -e "install.packages('lsa')"
RUN r -e "install.packages('clValid')"
RUN r -e "install.packages(c('devtools','colorspace'))"
RUN r -e "devtools::install_github('exaexa/scattermore')"

# CMD ["/init"]

# RUN echo 'source("/RInterface/setup.R")' >> /usr/local/lib/R/etc/Rprofile.site
