reticulate::install_miniconda()
reticulate::use_condaenv("/home/rstudio/.local/share/r-miniconda/envs/r-reticulate/bin/python")
reticulate::py_install('tensorflow', pip=TRUE, pip_options='--ignore-installed certifi')
reticulate::py_install('/backend', pip=TRUE)
np <- reticulate::import("numpy")
CellCNN <- reticulate::import("CellCNN")

# For some ungodly reason this import works differently on Windows and
# unix machines????? So... This solves it :-)
tryCatch(rscripts <- reticulate::import("CellCNN.rscripts"),
         error=function(cond){rscripts <- reticulate::import("CellCNN.CellCNN.rscripts")})

source('/RInterface/flowCIPHE_enrich.R')
source('/RInterface/CellCnnData.R')
source('/RInterface/CellCnnFolder.R')
source('/RInterface/CellCnnAnalysis.R')
source('/RInterface/result_visualisation.R')
source('/RInterface/quick_run.R')