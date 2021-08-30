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

get_default_labels <- function(data_table)
{
  labels <- data_table[,c(2,4,5,8)]
  labels[,1] <- ifelse(labels[,2]=="1c",yes = 0, no = 1)
  labels[is.na(labels)] <- -1
  
  labels_desc <<- list()
  labels_desc[[1]] <<- "pT [1c, 2/3]"
  labels_desc[[2]] <<- "pos..LN.removed.LN"
  labels_desc[[3]] <<- "ki.67"
  labels_desc[[4]] <<- "age"
  class_description <<- c(2L,0L,0L,0L)
  labels
}

train_model <- function(data,
                        labels,
                        class_description,
                        multicell_size = 1000L,
                        amount=50000L,
                        test_amount = 10000L,
                        layers = list(16L),
                        epochs = 30L,
                        l1_weight = 5e-4,
                        patience = 5L,
                        NAME=NAME)
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
  model$save(paste(NAME, "/", "config.json", sep=""), paste(NAME, "/", "weights.h5", sep=""))
  sm <<- model$get_single_cell_model()
  results <- lapply(data, sm)
  results <<- lapply(results, np$array)
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
  unlist(useful)
}

get_responding <- function(NAME, results, file_names, useful)
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
  write.table(responding, paste(NAME, "responding.csv", sep=""), sep=",")
  responding
}

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

get_controls <- function(CONTROL_NAMES)
{
  control_files <<- lapply(CONTROL_NAMES, read.FCS)
  control_data <<- lapply(control_files, get_values)
  control_results <- lapply(control_data, sm)
  control_results <<- lapply(control_results, np$array)
}

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

basic_train <- function(NAME=NULL,
                        CHANNELS_FILE=NULL,
                        DATA_FILE=NULL,
                        FILES_SUFFIX=NULL,
                        CONTROL_NAMES=NULL,
                        multicell_size = 1000L,
                        amount=50000L,
                        test_amount=10000L,
                        layers=list(64L, 64L, 16L),
                        epochs=30L,
                        l1_weight=5e-4,
                        patience=5)
{
  if (is.null(NAME)) NAME <- gsub("[ :]", "_", Sys.time())
  if (is.null(CHANNELS_FILE)) CHANNELS_FILE <- "BC_LiveCells.txt"
  if (is.null(DATA_FILE)) DATA_FILE <- "BCa_cohort_souceklab.txt"
  if (is.null(FILES_SUFFIX)) FILES_SUFFIX <- "_LiveCellsvaevictis_b.fcs"
  if (is.null(CONTROL_NAMES)) CONTROL_NAMES <- c(list.files(pattern="PBMC.*vaevictis_b.fcs"),
                                                 list.files(pattern="MDA.*vaevictis_b.fcs"))
  data <- load_data(CHANNELS_FILE, DATA_FILE, FILES_SUFFIX)
  labels <- get_default_labels(data_table)
  system(paste("mkdir", NAME))
  train_model(data = data,
              labels = labels,
              class_description = class_description,
              multicell_size = multicell_size,
              amount = amount,
              test_amount = test_amount,
              layers = layers,
              epochs = epochs,
              l1_weight = l1_weight,
              patience = patience,
              NAME=NAME)
  useful <- get_useful(results)
  responding <- get_responding(NAME,
    results = results, file_names = file_names, useful = useful)
  get_controls(CONTROL_NAMES)
  save_results(NAME, fcs_files, file_names, useful, results)
  save_results(NAME, control_files, CONTROL_NAMES, useful, control_results)
  model
}
