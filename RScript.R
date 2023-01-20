mydata <- read.csv("D449.csv")
#mydata$Day <- as.Date(mydata$Day, format="%d/%m/%Y")
View(mydata)
daily_ts <- ts(mydata$Micro, frequency = 365, start = c(2001,1,3))
plot(daily_ts)

#rolling origins definition
# Set horizon and number of rolling origins 
h <- 100
origins <- 299
daily_ts_length <- length(daily_ts)
#setting test and train length
train_length <- daily_ts_length - h - origins + 1
test_length <- h + origins - 1
daily_ts_train <- ts(daily_ts[1:train_length], frequency=frequency(daily_ts), start=start(daily_ts))
daily_ts_test <- daily_ts[(train_length+1):daily_ts_length]
#Matrix for holdout and forecasts
daily_ts_forecasts <- matrix(NA, nrow=origins, ncol=h)
daily_ts_holdout <- matrix(NA, nrow=origins, ncol=h)
colnames(daily_ts_forecasts) <- paste0("horizon",c(1:h))
rownames(daily_ts_forecasts) <- paste0("origin",c(1:origins))
dimnames(daily_ts_holdout) <- dimnames(daily_ts_forecasts)
for(i in 1:origins){
  # Create a ts object out of the daily_ts data
  our_train_set <- ts(daily_ts[1:(train_length+i-1)],
                      frequency=frequency(daily_ts),
                      start=start(daily_ts))
  # Write down the holdout values from the test set
  daily_ts_holdout[i,] <- daily_ts_test[i-1+(1:h)]
  # Produce forecasts and write them down
  daily_ts_forecasts[i,] <- forecast(ets(our_train_set,"ANN"),h=h)$mean
}
View(daily_ts_forecasts)
View(daily_ts_holdout)
#Matrix for error
error <- rowMeans(abs(daily_ts_holdout - daily_ts_forecasts))
error
min(error)

#splitting the series
# Find the total number of observations 
daily_ts_length <- length(daily_ts)
# Write down size of training set
train_length <- 3183
# And the forecasting horizon
h <- 398
# Create the training set
train <- ts(daily_ts[1:train_length], frequency=365, start = c(2001,1,3))
plot(train)
# Create the test set
test <- ts(daily_ts[(train_length+1):daily_ts_length],frequency=365, start = c(2009,265))

#Naive method
naive_method <- naive(train, h=h)
naive_forecast <- naive_method$mean
plot(naive_method)
Naive_RMSE <- sqrt(mean((test-naive_forecast)^2))
Naive_RMSE
Naive_MAE <- mean(abs(test-naive_forecast))
Naive_MAE

tsdisplay(daily_ts)
#test stationarity
kpss.test(train)
# you can reject the Null Hypothesis of the data being stationary. 
#So KPSS test tells us that we have enough evidence to say that data is not stationary.
adf.test(train)
#The p-value displayed is approximately equal to 0.5416 
#which means that we fail to reject the Null Hypothesis of non-stationarity at 5%.

#Calculating first difference since ts is not stationary.
diff_train <- diff(train)
kpss.test(diff_train)
adf.test(diff_train)
#the test shows that data is stationary and now we can check for ACF and PACF
tsdisplay(diff_train)
#Since one peak in PACF so we will start with ARI(1,1) i.e. Arima(1,1,0)
first_fit <- Arima(train, order=c(1,1,1))
# Plot series, ACF and PACF of the residuals
tsdisplay(residuals(first_fit))
first_fit
#since ACF & PACF both don't have anymore peaks, this looks like a best fit so we will proceed with this.
#We have found the best model and we can now forecast using this.
auto.arima(train)  #ARIMA(1,1,0)
#using auto.arima also we get the same order for our best fit.

#Find best method via AIC
auto.arima(train, ic="aic") #ARIMA(1,1,0)
#Find best method with ADF Test
auto.arima(train, test="adf") #ARIMA(1,1,0) 

second_fit <- Arima(train, order=c(0,1,1))
tsdisplay(residuals(second_fit))
second_fit

third_fit <- Arima(train, order=c(1,1,0))
third_fit
tsdisplay(residuals(third_fit))
plot(forecast(third_fit, h=398))


fourth_fit <- Arima(train, order = c(0,1,0))
fourth_fit

arima_fit_forecast <- forecast(third_fit, h=398)$mean
plot(forecast(first_fit, h=398))
acf(third_fit)

#Calculate errors
errors <- test - arima_fit_forecast
errors_ME <- mean(errors)
errors_MSE <- mean(errors ^ 2)
errors_RMSE <- sqrt(errors_MSE)
errors_MAPE <- 100 * mean(abs(errors)/test)

errors_RMSE
errors_MAPE

MAE <- mean(abs(errors))
MAE


#fitting simple moving average
SMA <- ma(train, order=1000, centre=FALSE)
plot(train)
lines(SMA, col = "red")
# Firstly we get rid of NA (‘‘Not Assigned’’) values: 
SMA_no_NAs <- SMA[!is.na(SMA)]
# Then form a forecast:
SMA3_forecast <- ts(rep(SMA_no_NAs[length(SMA_no_NAs)],398), frequency=365)
plot(test)
lines(SMA3_forecast, col = "red")

#fitting simple moving average
SMA <- ma(daily_ts, order=3, centre=FALSE)
# Firstly we get rid of NA (‘‘Not Assigned’’) values: 
SMA_no_NAs <- SMA[!is.na(SMA)]
# Then form a forecast:
SMA3_forecast <- ts(rep(SMA_no_NAs[length(SMA_no_NAs)],14), frequency=365, start = end(daily_ts))
plot(daily_ts)
lines(SMA3_forecast, col = "red")

#Calculate errors
SMA3_errors <- test - SMA3_forecast
SMA3_ME <- mean(SMA3_errors)
SMA3_MSE <- mean(SMA3_errors ^ 2)
SMA3_MAE <- mean(abs(SMA3_errors))
SMA3_MAPE <- 100 * mean(abs(SMA3_errors)/test)

#use MAE and RMSE for error comparison

#RMSE


#Naive method for forecast
naive_method <- naive(train, h=h)
naive_forecast <- naive_method$mean
plot(test)
plot(naive_forecast, col = "red")


#Exponential smoothing method
ETS_ANN <- ets(train, "ANN")
ETS_ANN
summary(ETS_ANN)
ETS_ANN_forecast <- forecast(ETS_ANN, h=398)$mean
plot(forecast(ETS_ANN, h=398))

ETS_MNN <- ets(train, "MNN")
ETS_MNN
summary(ETS_MNN)
ETS_MNN_forecast <- forecast(ETS_MNN, h=398)$mean
plot(forecast(ETS_MNN, h=398))

ETS_AAdN <- ets(train, model="AAN", damped=TRUE)
ETS_AAdN
summary(ETS_AAdN)
ETS_AAdN_forecast <- forecast(ETS_AAdN, h=398)$mean

ETS_MAdN <- ets(train, model = "MAN", damped = TRUE)
ETS_MAdN
summary(ETS_MAdN)
ETS_MAdN_forecast <- forecast(ETS_MAdN, h=398)$mean

es_ANN_initial <- es(daily_ts, model="ANN", initial=train[1], h=h, holdout=TRUE)
es_ANN_initial$accuracy
summary(es_ANN_initial)
es_ANN_initial_forecast <- forecast(es_ANN_initial, h=h)$mean

es_MNN_initial <- es(daily_ts, model="MNN", initial=train[1], h=398, holdout=TRUE)
es_MNN_initial$accuracy
summary(es_MNN_initial)
es_MNN_initial_forecast <- forecast(es_MNN_initial, h=398)$mean
plot(forecast(es_MNN_initial, h=398), main="ETS(MNN) with fixed seed")

#Calculate errors
errors <- test - es_MNN_initial_forecast
errors_ME <- mean(errors)
errors_MSE <- mean(errors ^ 2)
errors_RMSE <- sqrt(errors_MSE)
errors_MAPE <- 100 * mean(abs(errors)/test)

errors_RMSE
errors_MAPE

MAE <- mean(abs(errors))
MAE




# Fit SES with fixed initial seed
es_ANN_initial_1 <- es(daily_ts, model="ANN", initial=daily_ts[1], h=h, holdout=TRUE)
es_ANN_initial_1$accuracy
es_ANN_initial_1_forecast <- forecast(es_ANN_initial_1, h=h)$mean
plot(forecast(es_ANN_initial_1, h=h))
# Fit SES with optimised Seed
es_ANN_opt <- es(daily_ts, model="ANN", h=h, holdout=TRUE)
es_ANN_opt$accuracy
es_ANN_opt_forecast <- forecast(es_ANN_opt, h=h)$mean
plot(forecast(es_ANN_opt, h=h))

# Fit SES with optimised Seed (Benchmarking)
daily_ts_naive <- es(daily_ts, model="ANN", persistence=1, h=h, holdout=TRUE)
daily_ts_naive_forecast <- forecast(daily_ts_naive, h=h)$mean
plot(forecast(daily_ts_naive, h=h))

# Calculate an Optimized ETS Method using ets()
ets_ZZZ <- ets(train, model="ZZN")
ets_ZZZ
# Do the same using es()
es_ZZZ <- es(daily_ts, model="ZZZ")
es_ZZZ

plot(train, ylab= "Train and test")
lines(test, col = "red")


#Regression Models
lag_train <- Lag(train, -1)
D447_reg <- cbind(train, lag_train)
colnames(D447_reg)<- c("original", "L1")

fit1 <- lm(original~., data =D447_reg )
summary(fit1)

fit2 <- lm(original~.-1, data =D447_reg )
summary(fit2)

tsdisplay(fit1$residuals)

fit3 <- lm(log(original)~log(L1), data =D447_reg )
summary(fit3)

#Regression Models
lag_train <- Lag(train, -1)
trend <- c(1:daily_ts_length)
D447_reg <- cbind(daily_ts, trend)
colnames(D447_reg)<- c("original", "trend")

reg_train<- window(D447_reg, start= start(D447_reg), end= c(2009,263))
reg_test<-  window(D447_reg, start= c(2009,264), end= end(D447_reg))
View(reg_test)

fit1 <- lm(original~., data =reg_train )
summary(fit1)

fit2 <- lm(original~.-1, data =reg_train )
summary(fit2)

tsdisplay(fit1$residuals)

fit3 <- lm(original~.-trend, data =reg_train )
summary(fit3)

fit4 <-lm(original~1, data =reg_train)
summary(fit4)

forecast<- predict(fit4, reg_test)
plot(predict(fit2, reg_test))

RMS_error = sqrt(mean((reg_test[,1]- forecast)^2))
RMS_error

auto_model <- step(fit4, formula(fit1), direction="forward")
auto_model1 <- step(fit1, direction="both")
auto_model
auto_model1

#Extract Residuals 
fit2_resid <- residuals(fit2)
#We will also need fitted values for our analysis, which can be extracted using fitted(): #Extract Residuals
fit2_fitted <- fitted(fit2)
fit2_resid
#Plot Histogram 
hist(fit2_resid)

#QQ-Plot 
qqnorm(fit2_resid)
qqline(fit2_resid)

#Calculating final forecast using ARIMA(1,1,0)
final_fit <- Arima(daily_ts, order=c(1,1,0))
final_fit
final_forecast <- forecast(final_fit, h=14)$mean
final_forecast
plot(forecast(final_fit, h=14))

