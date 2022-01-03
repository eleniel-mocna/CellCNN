
CellCnnAnalysis <- R6::R6Class(
  "CellCnnAnalysis",
  inherit = CellCnnFolder,
  public = list(
    usefull = NULL,
    results = list(),
    model_name = NULL,
    print = function(){
      super$print()
      cat("- trained:", length(private$.trained_params)>0, "\n")
      return(invisible(self))
    },
    do_analysis = function(name=NULL,
                           multicell_size = 1000L,
                           amount=50000L,
                           test_amount=10000L,
                           layers=list(16L),
                           epochs = 30L,
                           l1_weight=0,
                           patience=5L,
                           clean_labels = TRUE,
                           cleanup=FALSE){
      if (cleanup) self$cleanup_analysis()
      if (clean_labels) self$clean_labels()
      if (is.null(name)){
        name = paste0("analysis_", gsub("[ :]", "_", Sys.time()))
      }
      else {
        name = gsub(" ", "_", name)
      }
      if (any(is.na(self$.labels))){
        stop("<NA> values in labels detected!")
      }
      self$model_name <- name
      path_to_analysis <- glue::glue("{private$.path}{.Platform$file.sep}{name}")
      dir.create(path_to_analysis)
      private$.trained_params = list(
        name = name,
        path_to_analysis = path_to_analysis,
        multicell_size = multicell_size,
        amount = amount,
        test_amount = test_amount,
        layers = layers,
        epochs = epochs,
        l1_weight = l1_weight,
        patience = patience
      )
      private$train_model()
      return(invisible(self))
    },
    
    cleanup_analysis = function(){
      unlink(private$.trained_params$path_to_analysis, recursive = TRUE)
      results = list()
      usefull = NULL
      model_name = NULL
      return(invisible(self))
    },
    
    load_model = function(){
      if (trained) {
        private$.model <- CellCNN$CellCNN$load()  
      }
      else {
        stop("This enviroment has not been trained, yet. :-(")
      }
      return(invisible(self))
    },
    
    predict_fcs_folder = function(folder,
                                  output_folder,
                                  pattern="*.fcs",
                                  transform_function=self$get_dato,
                                  keep_results = TRUE){
      if (is.null(self$usefull)){
        self$usefull <- 1:unlist(tail(private$.trained_params$layers, n=1))
      }
      self$results <- list()
      paths <- list.files(folder, pattern, full.names = TRUE)
      file_names <- list.files(folder, pattern, full.names = FALSE)
      dir.create(output_folder)
      for (i in 1:length(paths)) {
        fcs <- flowCore::read.FCS(paths[i])
        result <- np$array(private$.sm(transform_function(fcs)))
        for (j in self$usefull){
          fcs <- FlowCIPHE::enrich.FCS.CIPHE(fcs,
                                       result[,j],
                                       nw.names = list(paste(self$model_name, "F:", j)))
        }
        flowCore::write.FCS(fcs, paste0(normalizePath(output_folder),"/",file_names[i]))
        if (keep_results){
          self$results[[i]] <- result[,self$usefull]
        }
      }
    },
    
    #' Return a matrix of row vector filters.
    filters_values = function(){
      last_layer <- private$.model$layers[[length(private$.trained_params$layers)*2]]
      return(np$array(last_layer$weights[[1]])[1,,])
    },
    
    default_cluster_filters = function(k=0){
      f <- self$filters_values()
      distances <- -(lsa::cosine(f)-1)/2
      hcl <- hclust(as.dist(distances))
      if (k==0){
        res <- rep(0,10)
        for (i in 2:10){
          cluster <- cutree(hcl,k=i)
          res[i] <- clValid::dunn(as.dist(distances), cluster)
        }
        k <- which.max(res)
      }
      
      clustering <- cutree(hcl, k=k) # Change this to h=height.
      
      # Indices for the given cluster
      clusters <- list()
      
      # Which filter is representative of its given cluster (in cluster indexing)
      representative <- list() 
      for (i in 1:max(clustering)) {
        indices = (1:length(clustering))[clustering==i]
        clusters[[i]] = indices
        representative[[i]]= indices[which.min(colSums(as.matrix(distances[indices, indices])))]
      }
      self$usefull <- unlist(representative)
      return(invisible(self))
    }
    ),
  active = list(
    trained = function(value){
      if (missing(value)){
        return(length(private$.trained_params)>0)
      }
      else {
        stop("Cannot overwrite `trained` attribute!")
      }
    }
  ),
  private = list(
    .model = NULL,
    .sm = NULL,
    .trained_params = list(),
    .config_path = "",
    .weights_path = "",
    train_model = function(){
      model <- rscripts$train_model(data = private$.data,
                                    labels = self$labels,
                                    multicell_size = private$.trained_params$multicell_size,
                                    amount = private$.trained_params$amount,
                                    test_amount = private$.trained_params$test_amount,
                                    layers = private$.trained_params$layers,
                                    epochs = private$.trained_params$epochs,
                                    classes = unlist(private$.label_description),
                                    l1_weight = private$.trained_params$l1_weight,
                                    patience = private$.trained_params$patience)
      private$.config_path <- paste0(private$.trained_params$path_to_analysis, "/config.json")
      private$.weights_path <- paste0(private$.trained_params$path_to_analysis, "/weights.h5")
      model$save(private$.config_path, private$.weights_path)
      private$.model <- model
      private$.sm <- model$get_single_cell_model()
      return(invisible(self))
    }
  )
)

#' Create an CellCnnAnalysis object from a child of CellCnnData
#' @param cellCnnData child of CellCnnData class, containing data for analysis.
CellCnnAnalysis_from_Data <- function(cellCnnData){
  original <- cellCnnData
  return(
    CellCnnAnalysis$new(original$data,
                        original$labels,
                        original$label_description,
                        original$path,
                        original$name)
  )
}