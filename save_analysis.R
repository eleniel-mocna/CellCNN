# This file contains the script `save_analysis`,
# it takes a trained enviroment as an input and saves the results
# using only the env$useful filters.
save_analysis <- function(env){
  save_res(env$NAME, env$original_env$fcs_paths, env$useful, env$results)
  # TODO: add all the nice graphs etc
}


save_res <- function(NAME, file_names, useful, results)
{
  # Rewrite this not to take the fcs files as an argument.
  for (k in 1:length(fcs)) {
    file_name <- paste(NAME, "/", file_names[[k]], sep="")
    print(paste("Processing file", file_name))
    
    for (i in useful){
      fcs[[k]] <- enrich.FCS.CIPHE(fcs[[k]],
                                   results[[k]][,i],
                                   nw.names = list(paste(NAME, "F:", i)))
    }
    write.FCS(fcs[[k]], file_name)
  }
}