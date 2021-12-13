distances <- -(cosine(f)-1)/2
hcl <- hclust(as.dist(distances))
clustering <- cutree(hcl, h=0.5) # Change this to h=height.

# Indices for the given cluster
clusters <- list()

# Which filter is representative of its given cluster (in cluster indexing)
representative <- list() 
for (i in 1:max(clustering)) {
  indices = (1:length(clustering))[clustering==i]
  clusters[[i]] = indices
  representative[[i]]= indices[which.min(colSums(as.matrix(distances[indices, indices])))]
}
res <- rep(0,10)
for (i in 2:10){
  cluster <- cutree(hcl,k=i)
  res[i] <- dunn(as.dist(distances), cluster)
}
k <- which.max(res)
