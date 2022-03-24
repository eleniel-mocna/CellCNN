#' This is an example script that works if workdirectory is

# This python stuff needs to be run here so it doesn't crash.
# If it still crashes, try again, it works 9/10 times.
{
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
}
#####################################
# Fill out the following variables: #
#####################################

path_to_analysis<-"<Enter your path here>"
analysis <- CellCnnAnalysis$new("")
layers <- list(128,64,16)
l1_weight <- 0
epochs = 10L
amount=5000L
test_amount = 1000L
learning_rate = 5e-3
k=250L

##############################################################
# If nothing more is needed, just run the code until the end #
##############################################################

fcs_names <- paste0("data/",rownames(analysis$labels),".fcs")
fcss <- lapply(fcs_names, flowCore::read.FCS)
name <- (glue::glue("analysis_{paste0(layers, collapse='-')}_{l1_weight}"))
print(name)
analysis$label_description<-analysis$label_description[c("ITP_like","Bronchiectasis"),]

#If you want to play with the labels, do it here.
analysis$do_analysis(layers = layers,
                     name = name,
                     l1_weight = l1_weight,
                     epochs = epochs,
                     amount=amount,
                     test_amount = test_amount,
                     learning_rate = learning_rate,
                     k=k)

# If you just want to load a trained model instead:
# analysis$load_model(name)

# Picking the interesting filters
analysis$default_cluster_filters()
analysis$usefull
analysis$plot_filters_dendro()

# Plotting correlation between filter response and labels
pdf(paste0(name,"/correlation.pdf"))
par(mfrow=c(3,3))
plot_correlation(fcss, analysis, names = fcs_names)
dev.off()

# Exporting data into new fcs files
dir.create(paste0(name,"/data"))
analysis$predict_fcs_folder("data", output_folder = paste0(name, "/data"))

# Show filter responses based on other dimensions:
pdf(paste0(name,"/cells.pdf"))
par(mfrow = c(3, 3))
visualise_filters(fcss[1:27], analysis, "<FL 5 Log>", "<FL 6 Log>", fcs_names,
                  trans_function = analysis$get_dato)
dev.off()