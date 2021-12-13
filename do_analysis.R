# This file contains the script `do_analysis`,
# It takes a prepared enviroment from `setup_env` and does a CelCNN analysis
# with given parameters.
# It returns an enviroment with:
# - a trained model
# - list of arrays [cell, filter] (every filter value for every cell in every file)
# - reference to the original enviroment

library(reticulate)
use_condaenv("/home/rstudio/.local/share/r-miniconda/envs/r-reticulate/bin/python")
CellCNN <- import("CellCNN")
rscripts <- import("CellCNN.rscripts")
np <- import("numpy")

library(flowCore)
#devtools::install_github("https://github.com/cipheLab/FlowCIPHE.git")
library(FlowCIPHE)
library(scattermore)
library(ggplot2)

do_analysis <- function(env,
                        NAME=NULL,
                        multicell_size = 1000L,
                        amount=50000L,
                        test_amount=10000L,
                        layers=list(16L),
                        epochs = 30L,
                        l1_weight=0,
                        patience=5L){
  result_env <- new.env()
  if (is.null(NAME)) result_env$NAME <- gsub("[ :]", "_", Sys.time())
  else result_env$NAME <- NAME
  dir.create(result_env$NAME)
  result_env$original_env <- env
  result_env$dir_path <- paste(env$path, result_env$NAME, sep="/")
  result_env$labels <- clean_labels(env$labels)
  class_description <- as.list(env$labels_description[,"type"])
  result_env$model <- rscripts$train_model(data = env$data,
                       labels = result_env$labels,
                       multicell_size = multicell_size,
                       amount = amount,
                       test_amount = test_amount,
                       layers = layers,
                       epochs = epochs,
                       classes = class_description,
                       l1_weight = l1_weight,
                       patience = patience)
  result_env$model$save(paste(result_env$NAME, "/", "config.json", sep=""), paste(result_env$NAME, "/", "weights.h5", sep=""))
  result_env$sm <- result_env$model$get_single_cell_model()
  results <- lapply(env$data, result_env$sm)
  result_env$results <- lapply(results, np$array)
  result_env$useful <- get_useful(result_env$results)
  return(result_env)
}

clean_labels <- function(labels){
  labels[is.na(labels)]<- -1
  return(labels)
}
get_useful <- function(results)
{
  useful <- list()
  for (i in 1:ncol(results[[1]])) {
    m <- 0
    for (k in 1:length(results)) {
      m <- m + max(results[[k]][,i])
    }
    if (m>0) {
      useful <- append(useful, i)
    }
  }
  return(unlist(useful))
}