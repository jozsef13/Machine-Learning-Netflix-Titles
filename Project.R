#Netflix Titles dataset

#load libraries
library(mlbench)
library(caret)
library(corrplot)
library(ltm)
library(klaR)

#define filename
filename <- "netflix_titles.csv"
#load CSV file from the local directory
NetflixTitles <- read.csv(filename, header = FALSE, na.strings = c("", "NA"))

#preview the first 5 rows
head(NetflixTitles)

#dimension of the dataset
dim(NetflixTitles)

#summarize attribute distribution for the entire dataset
summary(NetflixTitles)

#for each column, sums up all the empty values
colSums(is.na(NetflixTitles))

#Removing columns with NA values and description column because is no use
NetflixTitles <- NetflixTitles[, -c(1, 4:6, 11:12)]

#Removing rows with NA values
NetflixTitles <- NetflixTitles[complete.cases(NetflixTitles),]

#Renaming columns
names(NetflixTitles)[names(NetflixTitles) == "V2"] <- "type"
names(NetflixTitles)[names(NetflixTitles) == "V3"] <- "title"
names(NetflixTitles)[names(NetflixTitles) == "V7"] <- "date_added"
names(NetflixTitles)[names(NetflixTitles) == "V8"] <- "release_year"
names(NetflixTitles)[names(NetflixTitles) == "V9"] <- "rating"
names(NetflixTitles)[names(NetflixTitles) == "V10"] <- "duration"

NetflixTitles$rating[NetflixTitles$rating == "G" | NetflixTitles$rating == "PG" | NetflixTitles$rating == "TV-G" | NetflixTitles$rating == "TV-PG" |
                       NetflixTitles$rating == "TV-Y" | NetflixTitles$rating == "TV-Y7" | NetflixTitles$rating == "TV-Y7-FV" 
                     | NetflixTitles$rating == "PG-13"] <- "AP"
NetflixTitles$rating[NetflixTitles$rating == "NC-17" | NetflixTitles$rating == "NR" | NetflixTitles$rating == "R" | NetflixTitles$rating == "TV-MA" |
                       NetflixTitles$rating == "UR" | NetflixTitles$rating == "TV-14"] <- "Adult"

#Removing unnecessary row(first row)
NetflixTitles <- NetflixTitles[-1,]

#Searching and removing duplicates
NetflixTitles[duplicated(NetflixTitles[, 1:6]), ]
NetflixTitles <- NetflixTitles[!duplicated(NetflixTitles[, 1:6]), ]

#Split out validation dataset
# create list of 70% of the rows in the original dataset we can use for training
set.seed(7)
validation_index <- caret::createDataPartition(NetflixTitles$rating, p = 0.7, list = FALSE)

#select 30% of the data for validation
validationDataset <- NetflixTitles[-validation_index,]

#use the remaining 70% of the data for training and testing the models
trainingDataset <- NetflixTitles[validation_index,]

#Summarize data
#dimension of the dataset
dim(trainingDataset)
summary(trainingDataset)

#list types for each attribute
sapply(trainingDataset, class)
head(trainingDataset, n = 10)

#for each column, sums up all the empty values
colSums(is.na(trainingDataset))

# convert data types
trainingDataset[,4] <- as.numeric(trainingDataset[,4])
trainingDataset[, 'date_added'] <- as.Date(trainingDataset[, 'date_added'], format = '%B %d, %Y')
trainingDataset[, 'type'] <- factor(trainingDataset[, 'type'], labels = c("Movie", "TV Show"))
trainingDataset[, 'rating'] <- factor(trainingDataset[, 'rating'], labels = c("Adult", "AP"))

#list types for each attribute
sapply(trainingDataset, class)

#Removing empty values created after conversion
colSums(is.na(trainingDataset))
trainingDataset <- trainingDataset[complete.cases(trainingDataset),]

#Correlations
biserial.cor(trainingDataset$release_year, trainingDataset$type)
cor(trainingDataset$release_year, as.numeric(trainingDataset$date_added, format = "%Y-%m-%d"))

#Plots for some attributes
ggplot(trainingDataset, aes(x=type, fill = type)) + geom_bar(show.legend=T)
ggplot(trainingDataset, aes(date_added, fill = date_added)) + geom_bar(show.legend=T)
ggplot(trainingDataset, aes(x=rating, fill = rating)) + geom_bar(show.legend=T)
ggplot(trainingDataset, aes(release_year, fill = type)) + geom_bar(show.legend=T)
ggplot(trainingDataset, aes(rating, fill = type)) + geom_bar(show.legend=T)

#Boxplot graph
ggplot(trainingDataset, aes(type, release_year, fill = type)) + geom_boxplot() + theme_grey()
ggplot(trainingDataset, aes(rating, release_year, fill = rating)) + geom_boxplot() + theme_grey()

#Density plot graph
ggplot(trainingDataset, aes(x=release_year, fill = type)) + geom_density(show.legend=T)
ggplot(trainingDataset, aes(x=date_added)) + geom_density(show.legend=T)

#Scatterplot matrix
ggplot(trainingDataset, aes(x=release_year, y=date_added)) + geom_point(shape=18, color="blue")

#Histogram on release_year
hist(trainingDataset$release_year, freq = F, ann = F)
#Density plot on release_year
plot(density(trainingDataset$release_year))

#Piecharts
ggplot(trainingDataset, aes(x="", fill=type)) + geom_bar(position="fill", width=1) + coord_polar("y") + theme_void()
ggplot(trainingDataset, aes(x="", fill=rating)) + geom_bar(position="fill", width=1) + coord_polar("y") + theme_void()

#Evaluate Algorithms
str(trainingDataset)
#Using cross validation with Accuracy metric
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
metric <- "Accuracy"
# GLMNET
set.seed(7)
fit.glmnet <- train(rating~type+release_year, data=trainingDataset, method="glmnet", metric=metric, trControl=control)
# CART
set.seed(7)
fit.cart <- train(rating~type+release_year, data = trainingDataset, method = "rpart", metric = metric, trControl=control)
# kNN(k-Nearest Neighbour)
set.seed(7)
fit.knn <- train(rating~type+release_year, data = trainingDataset, method = "knn", metric = metric, trControl=control)
#Support Vector Machine
set.seed(7)
fit.svm <- train(rating~type+release_year, data = trainingDataset, method = "svmRadial", metric = metric, trControl=control)
#Linear Discriminant Analysis
set.seed(7)
fit.lda <- train(rating~type+release_year, data = trainingDataset, method = "lda", metric = metric, trControl=control)

results <- resamples(list(GLMNET=fit.glmnet, CART=fit.cart, KNN=fit.knn, SVM=fit.svm, LDA = fit.lda))

summary(results)
dotplot(results)

#Ensemble Methods
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
metric <- "Accuracy"
# Random Forest
set.seed(7)
fit.rf <- train(rating~type+release_year, data = trainingDataset, method="rf", metric=metric, trControl=control)
# Stochastic Gradient Boosting
set.seed(7)
fit.gbm <- train(rating~type+release_year, data = trainingDataset, method="gbm", metric=metric, trControl=control, verbose=FALSE)
# Compare algorithms
ensemble_results <- resamples(list(RF=fit.rf, GBM=fit.gbm))
summary(ensemble_results)
dotplot(ensemble_results)

#Best model
print(fit.rf)

# convert data types
validationDataset[,4] <- as.numeric(validationDataset[,4])
validationDataset[, 'date_added'] <- as.Date(validationDataset[, 'date_added'], format = '%B %d, %Y')
validationDataset[, 'type'] <- factor(validationDataset[, 'type'], labels = c("Movie", "TV Show"))
validationDataset[, 'rating'] <- factor(validationDataset[, 'rating'], labels = c("Adult", "AP"))

predictions <- predict(fit.rf, validationDataset[, c(1,4,5)])

print(predictions)
confusionMatrix(predictions, validationDataset$rating)

saveRDS(fit.rf, "MyFinalModel.rds")