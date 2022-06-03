reticulate::use_condaenv("/home/rstudio/.local/share/r-miniconda/envs/r-reticulate/bin/python")
np <- reticulate::import("numpy")
CellCNN <- reticulate::import("CellCNN")
tryCatch(rscripts <- reticulate::import("CellCNN.rscripts"),
            error=function(cond){rscripts <- reticulate::import("CellCNN.CellCNN.rscripts")})
source('/RInterface/CellCnnData.R')
source('/RInterface/CellCnnFolder.R')
source('/RInterface/CellCnnAnalysis.R')
source('/RInterface/result_visualisation.R')
#####################################
# Fill out the following variables: #
#####################################
quick_run <- function(path,
                    csv=FALSE,
                    l1_weight=0,
                    epochs=10L,
                    amount=50000L,
                    learning_rate=5e-3,
                    k=250L){
    read_csv<-csv
    analysis <- CellCnnAnalysis$new(path, read_csv=read_csv)
    test_amount = as.integer(amount/5)
    

    if (read_csv){
        fcs_names <- paste0(path, "/","data/",rownames(analysis$labels),".csv")
        fcss<-parallel::mclapply(fcs_names, function(x){
            flowCore::flowFrame(as.matrix(read.csv(x, check.names = FALSE)))
        })
    } else {
        fcs_names <- paste0(path, "/","data/",rownames(analysis$labels),".fcs")
        fcss <- parallel::mclapply(fcs_names, flowCore::read.FCS)   
    }
    
    layers <- list(128)
    name <- (glue::glue("ordinal_results_{paste0(layers, collapse='-')}_{l1_weight}"))
    print(name)

    analysis$do_analysis(layers = layers,
                        name = name,
                        l1_weight = l1_weight,
                        epochs = epochs,
                        amount=amount,
                        test_amount = test_amount,
                        learning_rate = learning_rate,
                        k=k)

    analysis$default_cluster_filters(0)
    if (length(analysis$usefull)>5){
        analysis$default_cluster_filters(5)
    }

    pdf(paste0(path, "/", name,"/correlation.pdf"))
        par(mfrow=c(3,3))
        plot_correlation(fcss, analysis, names = fcs_names)
    dev.off()

    dir.create(paste0(path, "/", name,"/data"))
        if (read_csv){
            analysis$predict_csv_folder(paste0(path, "/","data"), output_folder = paste0(path, "/", name, "/data"))
        } else {
            analysis$predict_fcs_folder(paste0(path, "/","data"), output_folder = paste0(path, "/", name, "/data"))
        }



    layers <- list(128,64,32)
    name <- (glue::glue("ordinal_results_{paste0(layers, collapse='-')}_{l1_weight}"))
    print(name)

    analysis$do_analysis(layers = layers,
                        name = name,
                        l1_weight = l1_weight,
                        epochs = epochs,
                        amount=amount,
                        test_amount = test_amount,
                        learning_rate = learning_rate,
                        k=k)

    analysis$default_cluster_filters(0)
    if (length(analysis$usefull)>5){
        analysis$default_cluster_filters(5)
    }

    pdf(paste0(path, "/", name,"/correlation.pdf"))
        par(mfrow=c(3,3))
        plot_correlation(fcss, analysis, names = fcs_names)
    dev.off()

    dir.create(paste0(path, "/", name,"/data"))
        if (read_csv){
            analysis$predict_csv_folder(paste0(path, "/","data"), output_folder = paste0(path, "/", name, "/data"))
        } else {
            analysis$predict_fcs_folder(paste0(path, "/","data"), output_folder = paste0(path, "/", name, "/data"))
        }
}