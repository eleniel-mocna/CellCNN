#' This is an example script that works if workdirectory is 
#' "/home/rstudio/data/BCa_Souceklab/new_analysis"

# This python stuff needs to be run here so it doesn't crash.
# If it still crashes, try again, it works 9/10 times.
reticulate::use_condaenv("/home/rstudio/.local/share/r-miniconda/envs/r-reticulate/bin/python")
np <- reticulate::import("numpy")
CellCNN <- reticulate::import("CellCNN")
rscripts <- reticulate::import("CellCNN.rscripts")
library(FlowCIPHE) # This library doesn't work unless imported via library
source('~/data/git/CellCNN/CellCnnData.R')
source('~/data/git/CellCNN/CellCnnFolder.R')
source('~/data/git/CellCNN/CellCnnAnalysis.R')

# This will be moved into one file for sourcing, but reticulate and FlowCiphe
# still crash time from time and this way they work the best...


analysis <- CellCnnAnalysis$new(".")
analysis$labels <- analysis$labels[,c(1,2,4)]
analysis$do_analysis(layers = list(64,64,16), name = "testing")

analysis$default_cluster_filters(3)
analysis$predict_fcs_folder("data", "testing/results", keep_results = TRUE)

# results can now be accessed via analysis$results