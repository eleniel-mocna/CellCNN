# This file contains preprocessing of fcs data for CellCNN analysis
preprocess_env <- function(env, prepro_function){
  env$data <- lapply(env$data, prepro_function)
  env$controls_data <- lapply(env$data, prepro_function)
}

preprocess_arcsin <- function(data){
  get_values <- function(dato){
    return(asinh(dato/5))
  }
  return(lapply(data, get_values))
}
