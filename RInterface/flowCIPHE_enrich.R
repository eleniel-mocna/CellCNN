enrich.FCS.CIPHE <- function(original, new.column, nw.names=NULL){
  new_p <- flowCore::parameters(original)[1,]
  ## Now, let's change it's name from $P1 to $P26 (or whatever the next new number is)
  new_p_number <- as.integer(dim(original)[2]+1)
  # while(c(paste0("$P",
  # new_p_number))%in%row.names(pData(original@parameters))){ new_p_number <-
  # new_p_number+1 }
  rownames(new_p) <- c(paste0("$P", new_p_number))
  
  ## Now, let's combine the original parameter with the new parameter
  library('BiocGenerics') ## for the combine function
  allPars <-  BiocGenerics::combine(flowCore::parameters(original), new_p)
  
  ## Fix the name and description of the newly added parameter, say we want to be calling it cluster_id
  
  if(is.null(nw.names)){
    new_p_name <- "cluster"
  } else {
    new_p_name <- nw.names
  }
  
  allPars@data$name[new_p_number] <- new_p_name
  allPars@data$desc[new_p_number] <- new_p_name
  
  new_exprs <- cbind(original@exprs, new.column)
  colnames(new_exprs) <- c(colnames(original@exprs),new_p_name)
  
  new_kw <- original@description
  new_kw["$PAR"] <- as.character(new_p_number)
  new_kw[paste0("$P",as.character(new_p_number),"N")] <- new_p_name
  new_kw[paste0("$P",as.character(new_p_number),"S")] <- new_p_name
  new_kw[paste0("$P",as.character(new_p_number),"E")] <- "0,0"
  new_kw[paste0("$P",as.character(new_p_number),"G")] <- "1"
  new_kw[paste0("$P",as.character(new_p_number),"B")] <- new_kw["$P1B"]
  new_kw[paste0("$P",as.character(new_p_number),"R")] <- new_kw["$P1R"]
  new_kw[paste0("flowCore_$P",as.character(new_p_number),"Rmin")] <- new_kw["flowCore_$P1Rmin"]
  new_kw[paste0("flowCore_$P",as.character(new_p_number),"Rmax")] <- new_kw["flowCore_$P1Rmax"]
  
  ## Now, let's just combine it into a new flowFrame
  new_fcs <- new("flowFrame", exprs=new_exprs, parameters=allPars, description=new_kw)
  
  return(new_fcs)
}