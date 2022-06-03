#' This is an example script that works if workdirectory is

# This python stuff needs to be run here so it doesn't crash.
# If it still crashes, try again, it works 9/10 times.
{
    reticulate::use_condaenv("/home/rstudio/.local/share/r-miniconda/envs/r-reticulate/bin/python")
    np <- reticulate::import("numpy")
    CellCNN <- reticulate::import("CellCNN")
    tryCatch(rscripts <- reticulate::import("CellCNN.rscripts"),
             error=function(cond){rscripts <- reticulate::import("CellCNN.CellCNN.rscripts")})
    source('/RInterface/CellCnnData.R')
    source('/RInterface/CellCnnFolder.R')
    source('/RInterface/CellCnnAnalysis.R')
    source('/RInterface/result_visualisation.R')
}
#####################################
# Fill out the following variables: #
#####################################

path_to_analysis<-"."
read_csv <- FALSE # Is the data provided as a csv? (no: fcs)
analysis <- CellCnnAnalysis$new(path_to_analysis, get_dato=function(x)CellCnnFolder$public_methods$get_dato(fun=function(x) asinh(x/150)))
layers <- list(128)
l1_weight <- 0
epochs = 10L
amount=50000L
test_amount = 1000L
learning_rate = 5e-3
k=250L
name <- (glue::glue("results_{paste0(layers, collapse='-')}_{l1_weight}"))
print(name)

##############################################################
# If nothing more is needed, just run the code until the end #
##############################################################

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
analysis$default_cluster_filters(5)
analysis$usefull
analysis$plot_filters_dendro()

# Plotting correlation between filter response and labels
pdf(paste0(name,"/correlation.pdf"))
par(mfrow=c(3,3))
plot_correlation(fcss, analysis, names = fcs_names)
dev.off()

# Exporting data into new fcs files
output_data_folder = paste0(path_to_analysis, "/", name, "/data")
dir.create(output_data_folder)
if (read_csv){
    analysis$predict_csv_folder(path_to_analysis, "/data", output_folder = output_data_folder)
} else {
    analysis$predict_fcs_folder(path_to_analysis, "/data", output_folder = output_data_folder)
}


# Show filter responses based on other dimensions:

# pdf(paste0(name,"/cells.pdf"))
# par(mfrow = c(3, 3))
# visualise_filters(fcss, analysis, "PacificBlue-A", "PE-Cy7-A", fcs_names,
#                   trans_function = analysis$get_dato)
# dev.off()
