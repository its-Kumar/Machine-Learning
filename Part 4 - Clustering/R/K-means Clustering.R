# K-means Clustering

# Importing the mall dataset
dataset <- read.csv('Mall_Customers.csv')
X <- dataset[4:5]

# using the elbow method to find no of clusters
set.seed(6)
wcss <- vector()
for(i in 1:10) wcss[i] <- sum(kmeans(X, i)$withinss)
plot(1:10, wcss, type='b', main = paste("Clusters of clients"),
		 xlab = "Number of clusters", ylab = "WCSS")

# Applying k-means to mall dataset
set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)

# Visualizing the clusters
library(cluster)
clusplot(X,
				 kmeans$cluster, 
				 lines = 0,
				 shade = TRUE,
				 color = TRUE,
				 labels = 2,
				 plotchar = FALSE,
				 span = TRUE,
				 main = paste('Clusters of Clients'),
				 xlab = "Annual Income",
				 ylab = "Spanding Score")
