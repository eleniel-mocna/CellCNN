CellCnnAnalysis <- R6::R6Class(
  "CellCnnAnalysis",
  inherit = CellCnnData,
  public = list(
    results = list(),
    print = function(){
      cat("<CellCnnAnalysis>:", private$.name, "@", private$.path, "\n")
      cat("- n_samples: ", self$n_samples, "\n")
      cat("- n_channels:", self$n_channels, "\n")
      cat("- labels:", paste(colnames(private$.labels), collapse = ", "), "\n")
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
    },
    cleanup_analysis = function(){
      unlink(private$.trained_params$path_to_analysis, recursive = TRUE)
      results = list()
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
    .trained_params = list(),
    train_model = function(){
      model <- rscripts$train_model(data = private$.data,
                                    labels = private$.labels,
                                    multicell_size = private$.trained_params$multicell_size,
                                    amount = private$.trained_params$amount,
                                    test_amount = private$.trained_params$test_amount,
                                    layers = private$.trained_params$layers,
                                    epochs = private$.trained_params$epochs,
                                    classes = private$.label_description,
                                    l1_weight = private$.trained_params$l1_weight,
                                    patience = private$.trained_params$patience)
      model$save(paste0(private$.trained_params$path_to_analysis))
    }
  )
)