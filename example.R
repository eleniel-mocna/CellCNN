#' This is an example script that works if workdirectory is
#' "/home/rstudio/data/BCa_Souceklab/new_analysis"

# This python stuff needs to be run here so it doesn't crash.
# If it still crashes, try again, it works 9/10 times.

# reticulate::py_install("/home/rstudio/data/git/CellCNN", pip=TRUE)
reticulate::use_condaenv("/home/rstudio/.local/share/r-miniconda/envs/r-reticulate/bin/python")
np <- reticulate::import("numpy")
CellCNN <- reticulate::import("CellCNN")
rscripts <- reticulate::import("CellCNN.rscripts")

source('/home/data/Samuel_workdir/git/CellCNN/flowCIPHE_enrich.R')
source('~/data/git/CellCNN/CellCnnData.R')
source('~/data/git/CellCNN/CellCnnFolder.R')
source('~/data/git/CellCNN/CellCnnAnalysis.R')
source('~/data/git/CellCNN/result_visualisation.R')

analysis <- CellCnnAnalysis$new(".")
fcs_names <- paste0("data/",rownames(analysis$labels),".fcs")
fcss <- parallel::mclapply(fcs_names, flowCore::read.FCS)
layers <- list(64)
l1_weight <- 0
name <- (glue::glue("einlike-bronch{paste0(layers, collapse='-')}_{l1_weight}"))
print(name)


# Here you can adjust what labels etc. you want to use and keep them
# in analysis$label_description. The script  will filter all label columns not
# in the label_description table.
# If you want to reload labels/label_descriptions use:
#
# analysis$labels <- analysis$.__enclos_env__$private$get_labels("<PATH>")
# analysis$label_description <- (
#               analysis$.__enclos_env__$private$get_label_descr("<PATH>"))
# analysis$label_description<-analysis$label_description[1,]

analysis$do_analysis(layers = layers,
                     name = name,
                     l1_weight = l1_weight,
                     epochs = 10L,
                     amount=50000L,
                     test_amount = 10000L,
                     learning_rate = 5e-3,
                     k=250L)

#analysis$load_model(name)
analysis$default_cluster_filters(5)
analysis$usefull
analysis$plot_filters_dendro()
pdf(paste0(name,"/correlation.pdf"))
par(mfrow=c(3,3))
plot_correlation(fcss, analysis, names = fcs_names)#, filters=1:16)
dev.off()
dir.create(paste0(name,"/data"))
analysis$predict_fcs_folder("data", output_folder = paste0(name, "/data"))

pdf(paste0(name,"/cells.pdf"))
par(mfrow = c(3, 3))
visualise_filters(fcss[1:9], analysis, "<FL 5 Log>", "<FL 6 Log>", fcs_names,
                  trans_function = analysis$get_dato)
dev.off()
