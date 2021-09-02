NAME <- "test"
CHANNELS_FILE <- "BC_LiveCells.txt"
DATA_FILE <- "BCa_cohort_souceklab.txt"
FILES_SUFFIX <- "_LiveCellsvaevictis_b.fcs"
CONTROL_NAMES <- c(list.files(pattern="PBMC.*vaevictis_b.fcs"), list.files(pattern="MDA.*vaevictis_b.fcs"))


##### Setup #####
{
library(reticulate)
use_condaenv("/home/rstudio/.local/share/r-miniconda/envs/r-reticulate/bin/python")
CellCNN <- import("CellCNN")
library(flowCore)
#devtools::install_github("https://github.com/cipheLab/FlowCIPHE.git")
library(FlowCIPHE)
library(scattermore)
rscripts <- import("CellCNN.rscripts")
np <- import("numpy")
library(ggplot2)
}

##### Load data ##### 
load_data <- function(CHANNELS_FILE, DATA_FILE, FILES_SUFFIX)
{
  get_values <<- function(fcs)
    {
    ret <- exprs(fcs)[,channels[(channels[["use"]]==1),"name"]]
    asinh(ret/5)
    }
  channels <<- read.table(CHANNELS_FILE, header = TRUE)
  data_table <<- read.table(DATA_FILE, sep="\t", header=TRUE, na.strings = "x", dec=",")
  data_table <<- data_table[1:26,]
  file_names <<- paste(data_table[["X"]], FILES_SUFFIX, sep="")
  fcs_files <<- lapply(file_names, read.FCS)
  lapply(fcs_files, get_values)
}
data <- load_data(CHANNELS_FILE, DATA_FILE, FILES_SUFFIX)

###### Labels #######
{
  labels <- data_table[c(2)]
  labels[,1] <- ifelse(labels[,1]=="1c",yes = 0, no = 1)
  labels[is.na(labels)] <- -1
  
  labels_desc <- list()
  labels_desc[[1]] <- "pT [1c, 2/3]"
  class_description <- list(2L)
}

###### Training model and getting results ######
train_model <- function(data,
                        labels,
                        class_description,
                        multicell_size = 1000L,
                        amount=50000L,
                        test_amount = 10000L,
                        layers = list(16L),
                        epochs = 30L,
                        l1_weight = 5e-4,
                        patience = 5L)
{
  model <<- rscripts$train_model(data = data,
                              labels = labels,
                              multicell_size = multicell_size,
                              amount = amount,
                              test_amount = test_amount,
                              layers = layers,
                              epochs = epochs,
                              classes = class_description,
                              l1_weight = l1_weight,
                              patience = patience)
  system(paste("mkdir", NAME))
  model$save(paste(NAME, "/", "config.json", sep=""), paste(NAME, "/", "weights.h5", sep=""))
  sm <<- model$get_single_cell_model()
  results <- lapply(data, sm)
  results <<- lapply(results, np$array)
}
train_model(data, labels, class_description,1000L, 50000L, 10000L)

##### Processing results #####
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
  unlist(useful)
}
get_responding <- function(results, file_names, useful)
{
  get_percentage_responding <- function(x)
  {
    sum(x!=0)/length(x)*100
  }
  
  responding <- matrix(nrow = length(results), ncol=length(useful))
  colnames(responding) <- paste(NAME, "FILTER", useful)
  rownames(responding) <- file_names
  for (i in 1:length(useful)) {
    for (j in 1:length(file_names)) {
      responding[j,i] <- get_percentage_responding(results[[j]][,useful[i]])
    }
    
  }
  responding
}

useful <- get_useful(results)
responding <- get_responding(results, file_names, useful)

###### Visualisation ######
visualize <- function(results, files, filter_a, filter_b=NULL, filter_c=NULL)
{
  if (is.null(filter_b)) filter_b <- filter_a
  if (is.null(filter_c)) filter_c <- filter_a
  for (k in 1:length(results)){
    a <- results[[k]][,filter_a]
    a <- a + min(a)
    if(max(a)==0){maxa = 1}else{maxa=max(a)}
    a <- a/maxa
    a <- abs(1-a)
    
    b <- results[[k]][,filter_b]
    b <- b + min(b)
    if(max(b)==0){maxb = 1}else{maxb=max(b)}
    b <- b/maxb
    b <- abs(1-b)
    
    c <- results[[k]][,filter_c]
    c <- c + min(c)
    if(max(c)==0){maxc = 1}else{maxc=max(c)}
    c <- c/maxc
    c <- abs(1-c)
    
    color <- cbind(a, b, c)
    d <- exprs(files[[k]])
    scattermoreplot(d[,"vaevictis_1"],
                    d[,"vaevictis_2"],
                    col=rgb(color),
                    cex = 1.5)
    title(keyword(files[[k]])["$FIL"][[1]])
  } 
}
visualize(results, fcs_files, useful[1],useful[2],useful[3])
###### Control files ######
get_controls <- function(CONTROL_NAMES)
{
  control_files <<- lapply(CONTROL_NAMES, read.FCS)
  control_data <<- lapply(control_files, get_values)
  control_results <- lapply(control_data, sm)
  control_results <<- lapply(control_results, np$array)
}
get_controls(CONTROL_NAMES)
visualize(control_results, control_files, useful[1],useful[2],useful[3])

###### Saving files ######
save_results <- function(NAME, fcs, file_names, useful, results)
{
  for (k in 1:length(fcs)) {
    file_name <- paste(NAME, "/", file_names[[k]], sep="")
    print(paste("Processing file", file_name))
    
    for (i in useful){
      fcs[[k]] <- enrich.FCS.CIPHE(fcs[[k]],
                                   results[[k]][,i],
                                   nw.names = list(paste(NAME, "Filter", i)))
    }
    write.FCS(fcs_files[[k]], file_name)
  }
}

save_results(NAME, fcs_files, file_names, useful, results)
save_results(NAME, control_files, control_names, useful, control_results)

###### Getting filter weights  ######

# This helps print the right filters ([n] -> filter n + 1)
save_influence <- function(NAME, model)
{
  f <- file(paste(NAME, "/influence.txt", sep=""), open="wt")
  get_first_output_layer <- function(model)
  {
    for (i in length(model$layers):1) {
      if (model$layers[[i]]$name=="pooling") break
    
    }
    i+1
  }
  first_output_layer <- get_first_output_layer(model)
  for (i in 0:length(labels)-1) {
    for (k in useful){
      writeLines(paste("\n Filter: ", k, "Output", labels_desc[[i+1]],  "\t"), f)
      writeLines(paste("", model$layers[[first_output_layer+i]]$weights[[1]][k-1]$numpy()), f)
      
    }
  }
  close(f)
}
