reticulate::use_condaenv("/home/rstudio/.local/share/r-miniconda/envs/r-reticulate/bin/python")
CellCNN <- reticulate::import("CellCNN")
rscripts <- reticulate::import("CellCNN.rscripts")
np <- reticulate::import("numpy")

RCellCNN <- R6::R6Class(
  "RCellCNN",
  public = list(
    #' Initialize a CellCNN model with given hyper-parameters
    #' @param folder_path: string leading to the folder with this analysis
    #' @param ...: Hyper-parameters as described in the python implementation.
    initialize = function(folder_path,
                          multicell_size,
                          amount,
                          test_amount,
                          layers,
                          epochs,
                          l1_weight,
                          patience){
      private$.folder_path = normalizePath(folder_path)
      private$.multicell_size = multicell_size
      private$.amount = amount
      private$.test_amount = test_amount
      private$.layers = layers
      private$.epochs = epochs
      private$.l1_weight = l1_weight
      private$.patience = patience
      dir.create(private$.folder_path)
    },
    
    #' Print basic info about this object.
    print = function(){
      cat("<RCellCNN> model:\n")
      cat(" - folder_path:", private$.folder_path, "\n")
      cat(" - multicell_size:", private$.multicell_size, "\n")
      cat(" - amount:", private$.amount, "\n")
      cat(" - test_amount:", private$.test_amount, "\n")
      cat(" - layers: [", paste(private$.layers, collapse = ", "), "]\n")
      cat(" - epochs:", private$.epochs, "\n")
      cat(" - l1_weight:", private$.l1_weight, "\n")
      cat(" - patience:", private$.patience, "\n")
     
    },
    
    #' Train model with hyper parameters given to the constructor
    train_model = function(data,
                           labels,
                           label_description){
      private$.model <- rscripts$train_model(data = data,
                                             labels = labels,
                                             classes = label_description,
                                             multicell_size = private$.multicell_size,
                                             amount = private$.amount,
                                             test_amount = private$.test_amount,
                                             layers = private$.layers,
                                             epochs = private$.epochs,
                                             l1_weight = private$.l1_weight,
                                             patience = private$.patience)
    self$save_model()
    },
    
    save_model = function(){
      if (is.null(private$.model)){
        stop("There is no model to be saved!")
      }
      private$.model$save(paste0(private$.folder_path, "/config.json"),
                          paste0(private$.folder_path, "/weights.h5"))
    },
    
    load_model = function(){
      private$.model <- (
        CellCNN$CellCNN$load(paste0(private$.folder_path, "/config.json"),
                             paste0(private$.folder_path, "/weights.h5")))
      private$.sm <- private$.model$get_single_cell_model()
    },
    
    apply_model = function(data){
      results <- lapply(data, private$.sm)
      private$.last_results <- lapply(results, np$array)
      return(private$.last_results)
    }
  ),
  
  active = list(
    trained_epochs = function(value){
      if (missing(value)){
        return(private$.trained_epochs)
      }
      else {
        stop("Trained epochs cannot be modified!")
      }
    }
  ),
  private = list(
    .trained_epochs = 0,
    .folder_path = "",
    .multicell_size = 0L,
    .amount = 0L,
    .test_amount = 0L,
    .layers = list(),
    .epochs = 0L,
    .l1_weight = 0,
    .patience = 0L,
    .model = NULL,
    .sm = NULL
  )
)