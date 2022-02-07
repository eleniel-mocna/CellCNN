#' Visualise filters
#' @export
visualise_filters <- function(fcs_objects,
                              analysis,
                              dim1 = "vaevictis_1",
                              dim2 = "vaevictis_2",
                              names = NULL,
                              filters = NULL,
                              trans_function=function(x){flowCore::exprs(x)}) {
  if (is.null(filters)) {
    filters <- analysis$usefull
  }
  if (is.null(names)) {
    names <- lapply(fcs_objects, function(x) {
      keyword(x)["$FIL"]
    })
  }
  maxs <- double(max(filters)) + 1e-6
  for (fcs in fcs_objects) {
    this_dato <- analysis$get_dato(fcs)
    result <-
      np$array(analysis$.__enclos_env__$private$.sm(this_dato))
    for (i in filters) {
      maxs[i] = max(maxs[i], result[, i])
    }
  }
  results <- list()
  for (j in 1:length(fcs_objects)) {
    fcs <- fcs_objects[[j]]
    this_dato <- analysis$get_dato(fcs)
    results[[j]] <- np$array(analysis$sm(this_dato))
  }
  dim1_label = fcs_objects[[1]]@parameters@data[
    fcs_objects[[1]]@parameters@data[,"name"]==dim1,]
  dim2_label = fcs_objects[[1]]@parameters@data[
    fcs_objects[[1]]@parameters@data[,"name"]==dim2,]
  for (i in filters) {
    for (j in 1:length(fcs_objects)) {
      result <- results[[j]]
      fcs <- fcs_objects[[j]]
      color <- color_from_column(result[, i], maxs[i])
      o <- order(color)
      scattermore::scattermoreplot(cbind(
            trans_function(fcs)[, dim1],
            trans_function(fcs)[, dim2])[o, ],
         col = matrix(color[o], ncol = 1),xlab = dim1_label, ylab=dim2_label)
      # plot(p, xlab = dim1_label, ylab=dim2_label)
      
      title(glue::glue("{names[[j]]}, filter {i}."))
    }
  }
}

color_from_column <- function(x, maxx) {
  if (max(x) == 0) {
    x = x + 1
  }
  ret <- x - min(x)
  ret <- ret / maxx
  empty <- (ret == 0) * 0.8
  return(rgb(ret + empty, empty, empty, 1))
}

color_from_number <- function(x) {
  colorspace::diverge_hsv(100)[round(x * 100)]
}
#' Plot correlation between filters and labels
#' @export
plot_correlation <- function(fcs_objects,
                             analysis,
                             labels = NULL,
                             label_descr = NULL,
                             names = NULL,
                             filters = NULL)
{
  if (is.null(labels)) {
    if (is.null(label_descr)) {
      labels <- analysis$labels
      label_descr <- analysis$label_description
    }
    else{
      stop("If labels are provided, so need to be label descriptions!")
    }
  }
  else{
    warning(
      "Labels-label_descr compability not implemented yet! Unexpected things might happen :-("
    )
  }
  if (is.null(filters)) {
    filters <- analysis$usefull
  }
  if (is.null(names)) {
    names <-
      lapply(fcs_objects, function(x) {
        unlist(keyword(x)["$FIL"])
      })
  }
  results <- lapply(fcs_objects,
                    function(x) {
                      np$array(analysis$sm(analysis$get_dato(x)))
                    })
  result_corr <- lapply(results, function(x) {
    colSums(x != 0) / nrow(x)
  })
  result_corr <- do.call(rbind, result_corr)
  result_corr <- result_corr[, filters]
  colnames(result_corr) <- paste("F: ", filters)
  rownames(result_corr) <- names
  for (col in colnames(result_corr))
  {
    for (i in 1:dim(label_descr)[1]) {
      this_label_name <- rownames(label_descr)[i]
      this_label_descr <- label_descr[i, "type"]
      if (this_label_descr == 0) {
        lm1 <- lm(result_corr[, col] ~ labels[, this_label_name])
        tt <- paste("P=", round(summary(lm1)$coefficients[2, 4], 5))
        plot(
          result_corr[, col] ~ labels[, this_label_name],
          ylab = col,
          xlab = this_label_name,
          main = tt
        )
        abline(lm1)
      }
      else {
        boxplot(result_corr[, col] ~ labels[, this_label_name],
                ylab = col,
                xlab = this_label_name)
      }
      
    }
  }
}
