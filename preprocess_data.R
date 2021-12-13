# This file contains preprocessing of fcs data for CellCNN analysis
preprocess_env <- function(env, prepro_function){
  env$data <- lapply(env$data, prepro_function)
}

preprocess_arcsin <- function(dato){
  return(asinh(dato/5))
}
