default_cluster_filters <- function(k=0){
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
  return(unlist(representative))
}