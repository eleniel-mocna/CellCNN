# This file contains the script `setup_env`,
# which takes a path to a folder and prepares
# an enviroment for a CellCNN analysis
# based on data stored inside of it.
# This folder must contain the following files:
#  - labels.tsv
#  - labels_description.tsv
#  - channels.tsv
#  - controls.tsv
#  - data/<*>.fcs
#
# For the format of these files, see README

library(reticulate)
library(flowCore)
# This method wraps 
setup_env <- function(origin_folder){
  env <- new.env()
  env$path <- origin_folder
  env$labels_description_file <- glue::glue(
    "{origin_folder}/labels_description.tsv")
  env$labels_file <- glue::glue("{origin_folder}/labels.tsv")
  env$channels_file <- glue::glue("{origin_folder}/channels.tsv")
  env$data_folder <- glue::glue("{origin_folder}/data/")
  
  # TODO: Don't use this \/, instead write analyse with env function
  env$controls_file <- glue::glue("{origin_folder}/controls.tsv") 
  load_directors_into_env(env)
  load_data_into_env(env)
  return(env)
}

load_data_into_env <- function(env){
  get_values <- function(fcs){
    ret <- exprs(fcs)[,env$usable_channels]
    return(ret)
  }
  get_controls <- function(controls_file, path_to_root){
    controls_table <- read.table(controls_file, sep="\t", header = TRUE)
    controls_paths <- controls_table[,"name"]
    return(paths_to_data(paste(path_to_root,controls_paths, sep="/")))
  }
  paths_to_data <- function(paths){
    fcs_files <- lapply(paths, read.FCS)
    return(lapply(fcs_files, get_values))
  }
  env$fcs_paths <- glue::glue("{env$data_folder}{env$names}.fcs")
  env$data <- paths_to_data(env$fcs_paths)
  env$controls_data <- get_controls(env$controls_file, env$path)
}

load_directors_into_env <- function(env){
  env$labels_description <- load_labels(env$labels_description_file)
  labels <- load_labels(env$labels_file)
  env$labels <- labels[,colnames(labels) %in% env$labels_description[["name"]]]
  env$names <- labels[,"name"]
  env$fcs_paths <- glue::glue('{env$data_folder}{unlist(env$names)}.fcs')
  env$channels <- load_channels(env$channels_file)
  env$usable_channels <- env$channels[env$channels[["use"]]==1, "name"]
  load_labels <- function(labels_file){
    tab <- read.table(labels_file, sep="\t", header=TRUE)
    if (colnames(tab)[1]!="name"){
      warning("Labels file has an unexpected shape!")
    }
    return(tab)
  }
  load_labels_description <- function(labels_description_file){
    header_format <- list("name", "type")
    tab <- read.table(labels_description_file, sep="\t", header=TRUE)
    if(all(colnames(tab) == header_format))
    {  }
    else{
      warning("Labels description file has an unexpected shape!")
    }
    return(tab)
  }
  load_channels <- function(channels_file){
    tab <- read.table(channels_file, sep="\t", header=TRUE)
    if (all(c("use","name") %in% colnames(tab))){
      return(tab)
    }
    else{
      stop(glue::glue("Channels file expected to have 'name', 'use' columns, instead got [{paste(colnames(tab), collapse = ', ')}]"))
    }
  }
}