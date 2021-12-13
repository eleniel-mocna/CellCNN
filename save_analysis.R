# This file contains the script `save_analysis`,
# it takes a trained enviroment as an input and saves the results
# using only the env$useful filters.

#devtools::install_github("https://github.com/cipheLab/FlowCIPHE.git")
library(FlowCIPHE)
library(flowCore)

save_analysis <- function(env){
  result_paths <- glue::glue("{env$dir_path}/{env$original_env$names}.fcs")
  save_res(env$NAME, env$original_env$fcs_paths, result_paths, env$useful, env$results)
  # TODO: add all the nice graphs etc
}


save_res <- function(NAME, original_file_names, target_file_names, useful, results)
{
  fcs=lapply(original_file_names, read.FCS)
  for (k in 1:length(fcs)) {
    file_name <- target_file_names[k]
    print(paste("Processing file", file_name))
    
    for (i in useful){
      fcs[[k]] <- enrich.FCS.CIPHE(fcs[[k]],
                                   results[[k]][,i],
                                   nw.names = list(paste(NAME, "F:", i)))
    }
    write.FCS(fcs[[k]], file_name)
  }
}
plot_filters <- function(env, only_useful = TRUE){
  
}
