---
title: "\\vspace{-2cm} ML: Course Project"
author: "FJ Haran"
date: "2023-02-06"
output:
  html_document:
     keep_md: yes
---

\vspace{3pt}
# Executive Summary
The goal of this project is to create and select a Human Activity Recognition machine learning model that predicts what type of exercise an individual is performing. 
The bagging, boosting, and random forest models were trained and validated using a 70/30 split followed by a 5-fold cross validation. The random forest model had the lowest error (.006) and highest accuracy (.99). This model was then applied to the test data set and the results were uploaded as part of Quiz portion of the project. 

\vspace{3pt}
# Instructions
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website [ here](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).  

The goal of the project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

\vspace{-1cm}


\vspace{3pt}
## Download & load data

```r
download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv',
              destfile = './pml-training.csv', method = 'curl', quiet = T)

download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv',
              destfile = './pml-testing.csv', method = 'curl', quiet = T)

train <- read.csv("pml-training.csv", header=T, na.strings=c("","NA"))
test <- read.csv("pml-testing.csv", header=T, na.strings=c("","NA"))
```

\vspace{3pt}
## Pre-process data

```r
# remove NaNs
train <- train[, colSums(is.na(train)) == 0]
test <- test[, colSums(is.na(test)) == 0]

#remove non-numeric vars
train <- train[-c(1:7)]
test <- test[-c(1:7)]
test <- test[-c(53)]

#replace outcome character var with dummy coding
train[train$classe == "A",]$classe = 1
train[train$classe == "B",]$classe = 2
train[train$classe == "C",]$classe = 3
train[train$classe == "D",]$classe = 4
train[train$classe == "E",]$classe = 5

#create outcome var for training data set and delete var
train$classe <- as.factor(train$classe)
train_classe <- train[c(53)]
train <- train[-c(53)]

#Center and scale data
preProcValues <- preProcess(train, method = c("center", "scale"))
train <- predict(preProcValues, train)
test <- predict(preProcValues, test)

#Remove highly correlated predictors (>.8)
df2 = cor(train)
hc = findCorrelation(df2, cutoff=0.8) # put any value as a "cutoff" 
hc = sort(hc)
train_reduced = train[,-c(hc)]
test_reduced = test[,-c(hc)]

#Add back the outcome var to the training data set
new_train <- cbind(train_reduced, train_classe)
```

\vspace{3pt}
## Data analyses
### Split data into 70% training and 30% validation data sets

```r
set.seed(58789)
inTrain <- createDataPartition(y = new_train$classe, p = 0.7, list = F)
training <- new_train[inTrain, ]
validation <- new_train[-inTrain, ]
```

\vspace{3pt}
### Bagging  
1. Create bootstrap aggregation (bagging) prediction model

```r
fit_bag <- bagging(classe ~. , data = training)
pred_bag <- predict(fit_bag, validation)
confusionMatrix(as.factor(pred_bag$class), validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    1    2    3    4    5
##          1 1549  226   46   72   72
##          2   38  513   36   50  117
##          3   33  263  888  161  215
##          4   52  102   36  623   92
##          5    2   35   20   58  586
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7067          
##                  95% CI : (0.6949, 0.7183)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6273          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
## Sensitivity            0.9253  0.45040   0.8655   0.6463  0.54159
## Specificity            0.9012  0.94922   0.8617   0.9427  0.97606
## Pos Pred Value         0.7883  0.68037   0.5692   0.6884  0.83595
## Neg Pred Value         0.9681  0.87800   0.9681   0.9315  0.90432
## Prevalence             0.2845  0.19354   0.1743   0.1638  0.18386
## Detection Rate         0.2632  0.08717   0.1509   0.1059  0.09958
## Detection Prevalence   0.3339  0.12812   0.2651   0.1538  0.11912
## Balanced Accuracy      0.9133  0.69981   0.8636   0.7945  0.75882
```

2. Report accuracy metrics

```r
cat("The accuracy of the bagging model is:", 
    confusionMatrix(as.factor(pred_bag$class), validation$classe)$overall[1])
```

```
## The accuracy of the bagging model is: 0.706712
```


```r
cat("The Kappa statisic of the bagging model is:"
  , confusionMatrix(as.factor(pred_bag$class), validation$classe)$overall[2])
```

```
## The Kappa statisic of the bagging model is: 0.6272959
```

\vspace{3pt}
3. Perform 5-fold cross validation applied with bagging using 50 trees

```r
library(rpart)
fit_bag_cv <- bagging.cv(classe ~. , data = training, v = 5, mfinal = 50)
```

\vspace{3pt}
4. Calculate the out of sample error of the cross validation

```r
cat("The out of sample error estimate of the 5-folds cross validation procedure 
    applied with bagging is:", fit_bag_cv$error)
```

```
## The out of sample error estimate of the 5-folds cross validation procedure 
##     applied with bagging is: 0.2751693
```

\vspace{3pt}
### Boosting  
1. Create sequential ensemble (boosting) prediction model 

```r
fit_bst <- boosting(classe ~. , data = training)
pred_bst <- predict.boosting(fit_bst, newdata = validation, type = "class")
confusionMatrix(as.factor(pred_bst$class), validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    1    2    3    4    5
##          1 1637   92    1   16    4
##          2    9  932   65    7   20
##          3   13   92  937   64   43
##          4   11   19   21  849   30
##          5    4    4    2   28  985
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9074          
##                  95% CI : (0.8997, 0.9147)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8827          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
## Sensitivity            0.9779   0.8183   0.9133   0.8807   0.9104
## Specificity            0.9732   0.9787   0.9564   0.9835   0.9921
## Pos Pred Value         0.9354   0.9022   0.8155   0.9129   0.9629
## Neg Pred Value         0.9911   0.9573   0.9812   0.9768   0.9800
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2782   0.1584   0.1592   0.1443   0.1674
## Detection Prevalence   0.2974   0.1755   0.1952   0.1580   0.1738
## Balanced Accuracy      0.9755   0.8985   0.9348   0.9321   0.9512
```

\vspace{3pt}
2. Report accuracy metrics for the boosting model

```r
cat("The accuracy of the boosting model is:"
  , confusionMatrix(as.factor(pred_bst$class), validation$classe)$overall[1])
```

```
## The accuracy of the boosting model is: 0.9073917
```

\vspace{3pt}

```r
cat("The Kappa statisic of the boosting model is:"
  , confusionMatrix(as.factor(pred_bag$class), validation$classe)$overall[2])
```

```
## The Kappa statisic of the boosting model is: 0.6272959
```

\vspace{3pt}
3. Perform 5-fold cross validation applied with boosting using 50 trees

```r
fit_bst_cv <- boosting.cv(classe ~. , data = training, v = 5,  mfinal = 50)
```

```
## i:  1 Tue Feb  7 13:04:56 2023 
## i:  2 Tue Feb  7 13:06:31 2023 
## i:  3 Tue Feb  7 13:08:07 2023 
## i:  4 Tue Feb  7 13:09:44 2023 
## i:  5 Tue Feb  7 13:11:18 2023
```

\vspace{3pt}
4. Report out of sample error estimate for the cross validation

```r
cat("The out of sample error estimate of the 5-folds cross validation procedure 
    applied with boosting is:", fit_bst_cv$error)
```

```
## The out of sample error estimate of the 5-folds cross validation procedure 
##     applied with boosting is: 0.09288782
```

\vspace{3pt}
### Random forest (RF)
1. Create a RF model using the random forest package. 

```r
library(randomForest)
fit_rf <- randomForest(classe ~. , data = training, ntree=500, 
                       mtry = 6, nodesize = 5, importance = TRUE) 
pred_rf <- predict(fit_rf, validation)
confusionMatrix(pred_rf, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    1    2    3    4    5
##          1 1674    2    0    0    0
##          2    0 1131   10    0    0
##          3    0    6 1016   15    0
##          4    0    0    0  948    4
##          5    0    0    0    1 1078
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9935          
##                  95% CI : (0.9911, 0.9954)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9918          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
## Sensitivity            1.0000   0.9930   0.9903   0.9834   0.9963
## Specificity            0.9995   0.9979   0.9957   0.9992   0.9998
## Pos Pred Value         0.9988   0.9912   0.9797   0.9958   0.9991
## Neg Pred Value         1.0000   0.9983   0.9979   0.9968   0.9992
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1922   0.1726   0.1611   0.1832
## Detection Prevalence   0.2848   0.1939   0.1762   0.1618   0.1833
## Balanced Accuracy      0.9998   0.9954   0.9930   0.9913   0.9980
```

\vspace{3pt}
2. Report accuracy metric

```r
cat("The accuracy of the RF model is:"
  , confusionMatrix(pred_rf, validation$classe)$overall[1])
```

```
## The accuracy of the RF model is: 0.9935429
```

\vspace{3pt}

```r
cat("The Kappa statisic of the RF model is:"
  , confusionMatrix(pred_rf, validation$classe)$overall[2])
```

```
## The Kappa statisic of the RF model is: 0.991832
```

3. Report out-of-bag error metric

```r
cat("The OOB error of the RF model is:"
  , (1 - confusionMatrix(pred_rf, validation$classe)$overall[1]))
```

```
## The OOB error of the RF model is: 0.006457094
```


Note: In RFs, there is no need for cross validation or a separate test set to get an unbiased estimate of the test set error as it is estimated internally, during the run.

## Conclusion
The RF model achieved the best performance with an error rate of .6% and an
accuracy of 99%. 

# Quiz: Predict outcome ("classe") variable) for the test set
Apply the RF model to the test data set

```r
pred_rf_test <- predict(fit_rf, test_reduced)
pred_rf_test
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  2  1  2  1  1  5  4  2  1  1  2  3  2  1  5  5  1  2  2  2 
## Levels: 1 2 3 4 5
```
