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

get_default_labels <- function(data_table, indices)
{
  if (length(indices)>1) labels <- data_table[,indices]
  else labels <- data_table[indices]
  labels_desc <- list()
  class_description <- list()
  for (i in 1:length(indeces)) 
  {
    index <- indeces[[i]]
    switch (index,
      "1" = stop(),
      "2" = 
      {
        labels[,i] <- ifelse(labels[,i]=="1c",yes = 0, no = 1)
        labels_desc <- append(labels_desc, "pT [1c, 2/3]")
        class_description <- append(class_description, 2)
      },
      "3" = 
        {
          labels[,i][labels[,i]=="3a"] <- 3
          labels[,i][labels[,i]=="2a"] <- 2
          labels[,i][labels[,i]=="1a"] <- 1
          labels_desc <- append(labels_desc, "pN [0, 1a, 2a, 3a]")
          class_description <- append(class_description, 4)
        },
      "4" = 
        {
          labels_desc <- append(labels_desc, "pos_LN_removed_LN")
          class_description <- append(class_description, 0)
        },
      "5" = 
        {
          labels_desc <- append(labels_desc, "ki_67")
          class_description <- append(class_description, 0)
        },
      "6" = 
        {
          labels_desc <- append(labels_desc, "BRCA")
          class_description <- append(class_description, 2)
        },
      "7" = 
        {
          labels_desc <- append(labels_desc, "neoadj_therapy")
          class_description <- append(class_description, 2)
        },
      "8" = 
        {
          labels_desc <- append(labels_desc, "age")
          class_description <- append(class_description, 0)
        },
      "9" = 
        {
          labels_desc <- append(labels_desc, "_1_year_relapse")
          class_description <- append(class_description, 2)
        }
    )
  }
  labels_desc <- unlist(labels_desc)
  class_description <- unlist(class_description)
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
  write.table(responding, paste(NAME, "/responding.csv", sep=""), sep=",")
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
    a <- abs(0.9-a)
    
    b <- results[[k]][,filter_b]
    b <- b + min(b)
    if(max(b)==0){maxb = 1}else{maxb=max(b)}
    b <- b/maxb
    b <- abs(0.9-b)
    
    c <- results[[k]][,filter_c]
    c <- c + min(c)
    if(max(c)==0){maxc = 1}else{maxc=max(c)}
    c <- c/maxc
    c <- abs(0.9-c)
    
    color <- cbind(a, b, c)
    d <- exprs(files[[k]])
    scattermoreplot(d[,"vaevictis_1"],
                    d[,"vaevictis_2"],
                    col=rgb(color),
                    cex = 1.5)
    title(keyword(files[[k]])["$FIL"][[1]])
    
  } 
}
visualize_one <- function(results, files, filter)
{
  get_vae <- function(x)
  {
    exprs(x)[,c("vaevictis_1", "vaevictis_2")]
  }
  vae <- lapply(fcs_files, get_vae)
  vae <- do.call(rbind, vae)
  results <- do.call(rbind, results)
  a <- results[,filter]
  a <- a + min(a)
  if(max(a)==0){maxa = 1}else{maxa=max(a)}
  a <- a/maxa
  a <- a*0.9
  a <- abs(0.9-a)
  b <- a #lulz... xD
  c <- a*0 +0.9
  color <- cbind(c, a, b)
  scattermoreplot(vae[,1],
                  vae[,2],
                  col=rgb(color),
                  cex = 1.5)
  title(paste("FILTER", filter))
  
}

visualize_all <- function(all_results, useful, vae)
{
  for (f in useful) {
    response <- all_results[,f]
    #if(max(response)==0){maxr = 1}else{maxr=max(response)}
    #response <- response/maxr
    #response <- abs(1-response)
    response <- ifelse(response==0,0.9,0)
    scattermoreplot(vae[,1],
                    vae[,2],
                    col=rgb(response,response,response),
                    cex = 1.5)
    title(paste("FILTER", f))
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
                                   nw.names = list(paste(NAME, "F:", i)))
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
                        patience=5,
                        labels=c(1))
{
  if (is.null(NAME)) NAME <- gsub("[ :]", "_", Sys.time())
  if (is.null(CHANNELS_FILE)) CHANNELS_FILE <- "BC_LiveCells.txt"
  if (is.null(DATA_FILE)) DATA_FILE <- "BCa_cohort_souceklab.txt"
  if (is.null(FILES_SUFFIX)) FILES_SUFFIX <- "_LiveCellsvaevictis_b.fcs"
  if (is.null(CONTROL_NAMES)) CONTROL_NAMES <- c(list.files(pattern="PBMC.*vaevictis_b.fcs"),
                                                 list.files(pattern="MDA.*vaevictis_b.fcs"))
  data <- load_data(CHANNELS_FILE, DATA_FILE, FILES_SUFFIX)
  labels <- get_default_labels(data_table, labels)
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
  pdf(paste(NAME, "/output.pdf", sep=""))
  for (i in useful)
  {
    visualize_one(results, fcs_files, i)
  }
  dev.off()
  get_controls(CONTROL_NAMES)
  save_results(NAME, fcs_files, file_names, useful, results)
  save_results(NAME, control_files, CONTROL_NAMES, useful, control_results)
  model
}

load_model <- function(NAME)
{
  CellCNN$CellCNN$load(paste(NAME, "/config.json", sep=""),
                       paste(NAME, "/weights.h5", sep=""))
}
