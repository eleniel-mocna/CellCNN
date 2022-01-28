#' This is an example script that works if workdirectory is
#' "/home/rstudio/data/BCa_Souceklab/new_analysis"

# This python stuff needs to be run here so it doesn't crash.
# If it still crashes, try again, it works 9/10 times.
reticulate::use_condaenv("/home/rstudio/.local/share/r-miniconda/envs/r-reticulate/bin/python")
np <- reticulate::import("numpy")
CellCNN <- reticulate::import("CellCNN")
rscripts <- reticulate::import("CellCNN.rscripts")

#TODO: Vykopírovat enriche a bude
library(FlowCIPHE) # This library doesn't work unless imported via library
source('~/data/git/CellCNN/CellCnnData.R')
source('~/data/git/CellCNN/CellCnnFolder.R')
source('~/data/git/CellCNN/CellCnnAnalysis.R')
source('~/data/git/CellCNN/result_visualisation.R')

analysis <- CellCnnAnalysis$new(".")
fcs_names <- paste0("data/",rownames(analysis$labels),".fcs")
fcss <- lapply(fcs_names, flowCore::read.FCS)
for (layers in list(list(64), list(64,64,16),list(256,512,128,64))) {
  for (l1_weight in c(1e-5, 1e-4,1e-3)) {
    name <- (glue::glue("test_{paste0(layers, collapse='-')}_{l1_weight}"))
    print(name)
    analysis$do_analysis(layers = layers,
                         name = name,
                         l1_weight = l1_weight,
                         epochs = 10L,
                         amount=5000L,
                         test_amount = 1000L,
                         learning_rate = 5e-3)
    #analysis$load_model(name)
    analysis$default_cluster_filters()
    analysis$usefull
    analysis$plot_filters_dendro()
    pdf(paste0(name,"/correlation.pdf"))
    par(mfrow=c(3,3))
    plot_correlation(fcss, analysis, names = fcs_names)#, filters=1:64)
    dev.off()
    dir.create(paste0(name,"/data"))
    analysis$predict_fcs_folder("data", output_folder = paste0(name, "/data"))
    
    pdf(paste0(name,"/cells.pdf"))
    visualise_filters(fcss, analysis, "<FL 5 Log>", "<FL 8 Log>", fcs_names, trans_function = analysis$get_dato)
    dev.off()
  }
}
