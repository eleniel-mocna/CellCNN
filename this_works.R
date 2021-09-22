{library(reticulate)
use_condaenv("/home/rstudio/.local/share/r-miniconda/envs/r-reticulate/bin/python")
CellCNN <- import("CellCNN")
library(flowCore)
#devtools::install_github("https://github.com/cipheLab/FlowCIPHE.git")
library(FlowCIPHE)
library(scattermore)
rscripts <- import("CellCNN.rscripts")
np <- import("numpy")
library(ggplot2)
library(stringr)}
do_analysis <- function(NAME=NULL,
                        labels=c(2,4,5),
                        CHANNELS_FILE = "BC_LiveCells.txt",
                        DATA_FILE = "BCa_cohort_souceklab.txt",
                        FILES_SUFFIX = "_LiveCellsvaevictis_b.fcs",
                        CONTROL_NAMES = c(list.files(pattern="PBMC.*vaevictis_b.fcs"),
                                          list.files(pattern="MDA.*vaevictis_b.fcs")),
                        multicell_size = 1000L,
                        amount=50000L,
                        test_amount = 10000L,
                        layers = list(16L),
                        epochs = 30L,
                        l1_weight = 1e-6,
                        patience = 5L)
{
  env <- prepare_enviroment(NAME = NAME,
                     labels = labels,
                     CHANNELS_FILE = CHANNELS_FILE,
                     DATA_FILE = DATA_FILE,
                     FILES_SUFFIX = FILES_SUFFIX,
                     CONTROL_NAMES = CONTROL_NAMES)
  train_enviroment(env = env,
                   multicell_size = multicell_size,
                   amount = amount,
                   test_amount = test_amount,
                   layers = layers,
                   epochs = epochs,
                   l1_weight = l1_weight,
                   patience = patience)
  env
}

prepare_enviroment <- function(NAME=NULL,
                               labels=c(2,4,5),
                               CHANNELS_FILE = "BC_LiveCells.txt",
                               DATA_FILE = "BCa_cohort_souceklab.txt",
                               FILES_SUFFIX = "_LiveCellsvaevictis_b.fcs",
                               CONTROL_NAMES = c(list.files(pattern="PBMC.*vaevictis_b.fcs"),
                                                 list.files(pattern="MDA.*vaevictis_b.fcs"))
                               )
{
  env <- new.env()
  if (is.null(NAME)) env$NAME <- gsub("[ :]", "_", Sys.time())
  else env$NAME <- NAME
  env$CHANNELS_FILE <- CHANNELS_FILE
  env$DATA_FILE <- DATA_FILE
  env$FILES_SUFFIX <- FILES_SUFFIX
  env$CONTROL_NAMES <- CONTROL_NAMES
  env$label_indices <- labels
  load_data(env)
  get_labels(env)
  env
}
train_enviroment <- function(env,
                             multicell_size = 1000L,
                             amount=50000L,
                             test_amount = 10000L,
                             layers = list(16L),
                             epochs = 30L,
                             l1_weight = 1e-6,
                             patience = 5L)
{
  system(paste("mkdir", env$NAME))
  logs <- py_capture_output(
    train_model(env=env,
              multicell_size = multicell_size,
              amount = amount,
              test_amount = test_amount,
              layers = layers,
              epochs = epochs,
              l1_weight = l1_weight,
              patience = patience))
  logs <- str_extract_all(logs, "(val_[^-\b\n]*)|(Epoch [0-9]*\\/[0-9]*)")
  lapply(logs, write, paste(env$NAME, "/log.txt", sep=""), append=TRUE)
  #sink(paste(env$NAME, "/log.txt", sep=""), split = TRUE, append = TRUE)
  env$useful <- get_useful(env$results)
  if (length(env$useful)==0)
  {
    print("No filters functional! exiting")
    return(env)
  }
  env$responding <- get_responding(env)
  pdf(paste(env$NAME, "/output.pdf", sep=""))
  for (i in env$useful)
  {
    visualize_one(env, i)
  }
  dev.off()
  pdf(paste(env$NAME, "/filter_response.pdf", sep=""))
  boxplot(env$responding, names= env$useful)
  dev.off()
  get_controls(env)
  save_results(env)
  #sink()
}
load_data <- function(env)
{
  

  env$channels <- read.table(env$CHANNELS_FILE, header = TRUE)
  env$get_values <- function(fcs)
  {
    ret <- exprs(fcs)[,env$channels[(env$channels[["use"]]==1),"name"]]
    asinh(ret/5)
  }
  data_table <- read.table(env$DATA_FILE, sep="\t", header=TRUE, na.strings = "x", dec=",")
  env$data_table <- data_table[1:26,]
  env$file_names <- paste(env$data_table[["X"]], env$FILES_SUFFIX, sep="")
  env$fcs_files <- lapply(env$file_names, read.FCS)
  env$data <- lapply(env$fcs_files, env$get_values)
}

get_labels <- function(env)
{
  normal_data <- function(x)
  {
    x<-(x-mean(x, na.rm=TRUE))/sd(x, na.rm = TRUE)
    x-min(x, na.rm=TRUE)
  }
  #if (length(env$label_indices)>1) 
  labels <- env$data_table[unlist(env$label_indices)]
  #else labels <- env$data_table[,env$label_indices]
  labels_desc <- list()
  class_description <- list()
  for (i in 1:length(env$label_indices)) 
  {
    index <- env$label_indices[[i]]
    switch (index,
            "1" = stop(),
            "2" = 
              {
                labels[,i] <- ifelse(labels[,i]=="1c",yes = 0L, no = 1L)
                labels_desc <- append(labels_desc, "pT [1c, 2/3]")
                class_description <- append(class_description, 2L)
              },
            "3" = 
              {
                labels[,i][labels[,i]=="3a"] <- 3L
                labels[,i][labels[,i]=="2a"] <- 2L
                labels[,i][labels[,i]=="1a"] <- 1L
                labels_desc <- append(labels_desc, "pN [0, 1a, 2a, 3a]")
                class_description <- append(class_description, 4L)
              },
            "4" = 
              {
                labels[,i] <- normal_data(labels[,i])
                labels_desc <- append(labels_desc, "pos_LN_removed")
                class_description <- append(class_description, 0L)
              },
            "5" = 
              {
                labels[,i] <- normal_data(labels[,i])
                labels_desc <- append(labels_desc, "ki_67")
                class_description <- append(class_description, 0L)
              },
            "6" = 
              {
                labels_desc <- append(labels_desc, "BRCA")
                class_description <- append(class_description, 2L)
              },
            "7" = 
              {
                labels_desc <- append(labels_desc, "neoadj_therapy")
                class_description <- append(class_description, 2L)
              },
            "8" = 
              {
                labels[,i] <- normal_data(labels[,i])
                labels_desc <- append(labels_desc, "age")
                class_description <- append(class_description, 0L)
              },
            "9" = 
              {
                labels_desc <- append(labels_desc, "_1_year_relapse")
                class_description <- append(class_description, 2L)
              }
    )
  }
  labels[is.na(labels)] <- -1
  env$labels_desc <- labels_desc
  env$class_description <- class_description
  env$labels <- labels
}

train_model <- function(env,
                        multicell_size = 1000L,
                        amount=50000L,
                        test_amount = 10000L,
                        layers = list(16L),
                        epochs = 30L,
                        l1_weight = 5e-4,
                        patience = 5L)
{
  env$model <- rscripts$train_model(data = env$data,
                                 labels = env$labels,
                                 multicell_size = multicell_size,
                                 amount = amount,
                                 test_amount = test_amount,
                                 layers = layers,
                                 epochs = epochs,
                                 classes = env$class_description,
                                 l1_weight = l1_weight,
                                 patience = patience)
  env$model$save(paste(env$NAME, "/", "config.json", sep=""), paste(env$NAME, "/", "weights.h5", sep=""))
  env$sm <- env$model$get_single_cell_model()
  results <- lapply(env$data, env$sm)
  env$results <- lapply(results, np$array)
  
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

get_responding <- function(env)
{
  get_percentage_responding <- function(x)
  {
    sum(x!=0)/length(x)*100
  }
  
  responding <- matrix(nrow = length(env$results), ncol=length(env$useful))
  colnames(responding) <- paste(env$NAME, "FILTER", env$useful)
  rownames(responding) <- env$file_names
  for (i in 1:length(env$useful)) {
    for (j in 1:length(env$file_names)) {
      responding[j,i] <- get_percentage_responding(env$results[[j]][,env$useful[i]])
    }
    
  }
  write.table(responding, paste(env$NAME, "/responding.csv", sep=""), sep=",")
  responding
}

visualize <- function(env, filter_a, filter_b=NULL, filter_c=NULL)
{
  if (is.null(filter_b)) filter_b <- filter_a
  if (is.null(filter_c)) filter_c <- filter_a
  for (k in 1:length(env$results)){
    a <- env$results[[k]][,filter_a]
    a <- a + min(a)
    if(max(a)==0){maxa = 1}else{maxa=max(a)}
    a <- a/maxa
    a <- abs(0.9-a)
    
    b <- env$results[[k]][,filter_b]
    b <- b + min(b)
    if(max(b)==0){maxb = 1}else{maxb=max(b)}
    b <- b/maxb
    b <- abs(0.9-b)
    
    c <- env$results[[k]][,filter_c]
    c <- c + min(c)
    if(max(c)==0){maxc = 1}else{maxc=max(c)}
    c <- c/maxc
    c <- abs(0.9-c)
    
    color <- cbind(a, b, c)
    d <- exprs(env$fcs_files[[k]])
    scattermoreplot(d[,"vaevictis_1"],
                    d[,"vaevictis_2"],
                    col=rgb(color),
                    cex = 1.5)
    title(keyword(env$fcs_files[[k]])["$FIL"][[1]])
    
  } 
}
visualize_one <- function(env, filter)
{
  get_vae <- function(x)
  {
    exprs(x)[,c("vaevictis_1", "vaevictis_2")]
  }
  vae <- lapply(env$fcs_files, get_vae)
  vae <- do.call(rbind, vae)
  all_results <- do.call(rbind, env$results)
  a <- all_results[,filter]
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
get_controls <- function(env)
{
  env$control_files <- lapply(env$CONTROL_NAMES, read.FCS)
  env$control_data <- lapply(env$control_files, env$get_values)
  env$control_results <- lapply(env$control_data, env$sm)
  env$control_results <- lapply(env$control_results, np$array)
}

save_results <- function(env)
{
  save_res(env$NAME, env$fcs_files, env$file_names, env$useful, env$results)
  save_res(env$NAME, env$control_files, env$CONTROL_NAMES, env$useful, env$results)
  save_influence(env)
  
}

save_res <- function(NAME, fcs, file_names, useful, results)
{
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

save_influence <- function(env)
{
  f <- file(paste(env$NAME, "/influence.txt", sep=""), open="wt")
  get_first_output_layer <- function(model)
  {
    for (i in length(model$layers):1) {
      if (model$layers[[i]]$name=="pooling") break
      
    }
    i+1
  }
  first_output_layer <- get_first_output_layer(env$model)
  for (i in 0:(length(env$labels)-1)) {
    for (k in env$useful){
      writeLines(paste("\n Filter: ", k, "Output", env$labels_desc[[i+1]],  "\t"), f)
      writeLines(paste("", env$model$layers[[first_output_layer+i]]$weights[[1]][k-1]$numpy()), f)
      
    }
  }
  close(f)
}

load_model <- function(NAME)
{
  CellCNN$CellCNN$load(paste(NAME, "/config.json", sep=""),
                       paste(NAME, "/weights.h5", sep=""))
}

transfer_fcs <- function(source, target)
{
  source_files <- paste(source, "/", list.files(source, ".fcs"), sep="")
  source_fcs <- lapply(source_files,read.FCS)
  target_files <- paste(target, "/", list.files(source, ".fcs"), sep="")
  target_fcs <- lapply(target_files,read.FCS)
  source_filter_columns <- grep(source, colnames(source_fcs[[1]]))
  filter_names <- colnames(source_fcs[[1]])[source_filter_columns]
  for (k in 1:length(source_fcs))
  {
    for (i in 1:length(source_filter_columns))
    {
      filter <- source_filter_columns[i]
      filter_name <- filter_names[i]
      filter_results <- source_fcs[[k]]@exprs[,filter]
      target_fcs[[k]] <- enrich.FCS.CIPHE(target_fcs[[k]], filter_results, filter_name)
    }
    print(paste("Writing file", target_files[[k]]))
    write.FCS(target_fcs[[k]], target_files[[k]])
  }
}
labels <- list(2,4,5)
layers <- list(16L)
run5 <- function(name, labels, layers)
{
  for (i in 1:5) 
  {
    NAME <- paste(name,i, sep="")
    sink(paste("logs/", NAME, ".txt", sep=""), append=TRUE, split = TRUE)
    print(paste("#######Running", NAME))
    tryCatch(do_analysis(NAME = NAME,
                         labels = labels,
                         layers = layers,
                         l1_weight = 0),
             error=function(e)print(paste("ERROR:", e)),
             finally = {print(paste("ENDED", NAME))
               sink()})
  }
}

data_table <- read.table("../BCa_Souceklab/BCa_cohort_souceklab.txt", sep="\t", header = TRUE, dec = ",",na.strings = "x")
names <- data_table[c(1:26),1]
data_table <- data_table[c(1:26),]
rownames(data_table) <- data_table[,"X"]
data_table <- data_table[,2:10]
data_table[data_table=="x"] <- NA
file_names <- paste("BCa_souceklab_vaevictis/", names, "_LiveCellsvaevictis_b.fcs", sep="")
#file_names <- paste("../BCa_Souceklab/l1_0_ki67_S_nl_2/", names, "_LiveCellsvaevictis_b.fcs", sep="")
fcs_files <- lapply(file_names, read.FCS)
columns <- grep(pattern = " F: ", colnames(exprs(fcs_files[[2]])))
filter_values <- lapply(fcs_files, function(x)
  {
  res<-matrix(NA,nrow(x),length(columns))
  ss<-which(columns<=ncol(exprs(x)))
  
  res[,ss]<-  exprs(x)[,columns[ss]]
  res
  })

active <- lapply(filter_values, function(x){colSums(x!=0)/nrow(x)})
active <- do.call(rbind, active)
colnames(active)<-as.character(colnames(exprs(fcs_files[[3]]))[columns])
active_w_labels <- cbind(active, data_table)
rownames(active) <- names
#pdf("../Stemp.pdf")
par(mfrow = c(3, 3))
for (col in colnames(active)) {
  boxplot(active_w_labels[,col]~data_table[,"pT"], ylab=col, xlab="pT")
  lm1 <- lm(active_w_labels[,col]~data_table[,"pos..LN.removed.LN"])
  tt<-paste("P=",round(summary(lm1)$coefficients[2,4], 5))
  plot(active_w_labels[,col]~data_table[,"pos..LN.removed.LN"], ylab=col, xlab="pos",main=tt)
  abline(lm1)
  lm1 <- lm(active_w_labels[,col]~data_table[,"ki.67"])
  tt<-paste("P=",round(summary(lm1)$coefficients[2,4], 5))
  plot(active_w_labels[,col]~data_table[,"ki.67"], ylab=col, xlab="ki.67",main=tt)
  abline(lm1)
}
#dev.off()

#pdf("../absolutely_all.pdf")
par(mfrow = c(3, 3))
for (col in colnames(active)) {
  boxplot(active_w_labels[,col]~data_table[,"pT"], ylab=col, xlab="pT")
  boxplot(active_w_labels[,col]~data_table[,"pN"], ylab=col, xlab="pN")
  lm1 <- lm(active_w_labels[,col]~data_table[,"pos..LN.removed.LN"])
  tt<-paste("P=",round(summary(lm1)$coefficients[2,4], 5))
  plot(active_w_labels[,col]~data_table[,"pos..LN.removed.LN"], ylab=col, xlab="pos",main=tt)
  abline(lm1)
  lm1 <- lm(active_w_labels[,col]~data_table[,"ki.67"])
  tt<-paste("P=",round(summary(lm1)$coefficients[2,4], 5))
  plot(active_w_labels[,col]~data_table[,"ki.67"], ylab=col, xlab="ki.67",main=tt)
  abline(lm1)
  boxplot(active_w_labels[,col]~data_table[,"BRCA"], ylab=col, xlab="BRCA")
  boxplot(active_w_labels[,col]~data_table[,"neoadj..therapy"], ylab=col, xlab="neoadj..therapy")
  lm1 <- lm(active_w_labels[,col]~data_table[,"age"])
  tt<-paste("P=",round(summary(lm1)$coefficients[2,4], 5))
  plot(active_w_labels[,col]~data_table[,"age"], ylab=col, xlab="age",main=tt)
  abline(lm1)
  boxplot(active_w_labels[,col]~data_table[,"X1.year.relapse"], ylab=col, xlab="1 year relapse")
  boxplot(active_w_labels[,col]~data_table[,"run"], ylab=col, xlab="run")
}
dev.off()
boxplot(new_filter_values[,1]~new_data_table[,"pT"])
lm1 <- lm(new_filter_values[,1]~new_data_table[,"ki.67"])
tt<-paste("P=",round(summary(lm1)$coefficients[2,4], 5))
plot(new_filter_values[,1]~new_data_table[,"ki.67"],main=tt)
abline(lm1)
#run5("3_S_nl_", list(2,4,5), list(16L))file_names <- list.files()
#run5("3_M_nl_", list(2,4,5), list(128L,128L,128L,16L))
#run5("pT_S_nl_", list(2), list(16L))
#run5("pT_M_nl_", list(2), list(128L,128L,128L,16L))
#run5("pos_S_nl_", list(4), list(16L))
#run5("pos_M_nl_", list(4), list(128L,128L,128L,16L))
run5("l1_0_ki67_S_nl_", list(5), list(16L))
run5("l1_0_ki67_M_nl_", list(5), list(128L,128L,128L,16L))

