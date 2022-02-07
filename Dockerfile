FROM rocker/rstudio
LABEL NAME="cellcnn_eleniel" Version="1.0"
LABEL author="Samuel Soukup"
LABEL contact="soukup.sam(at)gmail.com"

# CellCnn folder should be mounted via -v

RUN mkdir /DockerBuildInfo
COPY Dockerfile /DockerBuildInfo

COPY README.md /README.md
COPY CellCNN /backend/CellCNN
COPY setup.py /backend/setup.py
COPY RInterface /RInterface

RUN apt-get update && \
    apt-get install zlib1g-dev && \
    apt-get clean

# RUN Rscript /RInterface/install.R
CMD ["/init"]

RUN echo 'source("/RInterface/setup.R")' >> /usr/local/lib/R/etc/Rprofile.site
