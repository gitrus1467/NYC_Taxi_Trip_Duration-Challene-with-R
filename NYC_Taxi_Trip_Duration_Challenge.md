# NYC_Taxi_Trip_Duration_Challenge
Assadullah_Samir  
5 November 2017  


# Introduction
This is a comprehensive Exploratory Data Analysis for the New York City Taxi Trip Duration competition with tidy R and ggplot2.
The goal of this playground challenge is to predict the duration of taxi rides in NYC based on features like trip coordinates or pickup date and time. The data comes in the shape of 1.5 million training observations (../input/train.csv) and 630k test observation (../input/test.csv). Each row contains one taxi trip.
In this notebook, we will first study and visualise the original data, engineer new features, and examine potential outliers. Then we add two external data sets on the NYC weather and on the theoretically fastest routes. We visualise and analyse the new features within these data sets and their impact on the target trip_duration values. Finally, we will make a brief excursion into viewing this challenge as a classification problem and finish this notebook with a simple XGBoost model that provides a basic prediction (final part under construction).


```r
library('ggplot2') # visualisation
library('scales') # visualisation
library('grid') # visualisation
library('RColorBrewer') # visualisation
library('corrplot') # visualisation
library('alluvial') # visualisation
library('dplyr') # data manipulation
library('readr') # input/output
library('data.table') # data manipulation
library('tibble') # data wrangling
library('tidyr') # data wrangling
library('stringr') # string manipulation
library('forcats') # factor manipulation
library('lubridate') # date and time
library('geosphere') # geospatial locations
library('leaflet') # maps
library('leaflet.extras') # maps
library('maps') # maps
library('xgboost') # modelling
library('caret') # modelling
library('plotly') #visualization
library ('gridExtra') # arrange plots
```
Load data


```r
train <- as.tibble(fread('C:/Users/assad/NYC_Kaggle/train.csv'))
```

File structure and content


```r
summary(train)
```

```
##       id              vendor_id     pickup_datetime    dropoff_datetime  
##  Length:1458644     Min.   :1.000   Length:1458644     Length:1458644    
##  Class :character   1st Qu.:1.000   Class :character   Class :character  
##  Mode  :character   Median :2.000   Mode  :character   Mode  :character  
##                     Mean   :1.535                                        
##                     3rd Qu.:2.000                                        
##                     Max.   :2.000                                        
##  passenger_count pickup_longitude  pickup_latitude dropoff_longitude
##  Min.   :0.000   Min.   :-121.93   Min.   :34.36   Min.   :-121.93  
##  1st Qu.:1.000   1st Qu.: -73.99   1st Qu.:40.74   1st Qu.: -73.99  
##  Median :1.000   Median : -73.98   Median :40.75   Median : -73.98  
##  Mean   :1.665   Mean   : -73.97   Mean   :40.75   Mean   : -73.97  
##  3rd Qu.:2.000   3rd Qu.: -73.97   3rd Qu.:40.77   3rd Qu.: -73.96  
##  Max.   :9.000   Max.   : -61.34   Max.   :51.88   Max.   : -61.34  
##  dropoff_latitude store_and_fwd_flag trip_duration    
##  Min.   :32.18    Length:1458644     Min.   :      1  
##  1st Qu.:40.74    Class :character   1st Qu.:    397  
##  Median :40.75    Mode  :character   Median :    662  
##  Mean   :40.75                       Mean   :    959  
##  3rd Qu.:40.77                       3rd Qu.:   1075  
##  Max.   :43.92                       Max.   :3526282
```

Reformating Data


```r
train <- train %>%
  mutate(pickup_datetime = ymd_hms(pickup_datetime),
         dropoff_datetime = ymd_hms(dropoff_datetime),
         vendor_id = factor(vendor_id),
         passenger_count = factor(passenger_count))
## Create hpick and Month col
train<-train %>%
  mutate(hpick = hour(pickup_datetime),
         Month = factor(month(pickup_datetime, label = TRUE)))
```

# Individual feature visualisations

Visualisations of feature distributions and their relations are key to understanding a data set, and they often open up new lines of inquiry. I always recommend to examine the data from as many different perspectives as possible to notice even subtle trends and correlations.

## Log distribution of the trip_duartions


```r
ggplot(train,aes(trip_duration,fill=vendor_id))+geom_histogram(bins = 100)+scale_x_log10()
```

![](fig/unnamed-chunk-6-1.png)<!-- -->
