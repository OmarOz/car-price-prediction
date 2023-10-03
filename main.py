import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


cars_dataSet = pd.read_csv(
    'C:/Users/oali2/OneDrive/Documents/growIntern tasks/car price prediction/archive/CarPrices.csv')


cars_dataSet.replace({'fueltype': {'gas': 0, 'diesel': 1}}, inplace=True)
cars_dataSet.replace({'enginelocation': {'front': 0, 'rear': 1}}, inplace=True)
cars_dataSet.replace(
    {'drivewheel': {'fwd': 0, 'rwd': 1, '4wd': 2}}, inplace=True)
cars_dataSet.replace(
    {'aspiration': {'std': 0, 'turbo': 1}}, inplace=True)
cars_dataSet.replace({'cylindernumber': {'two': 2, 'three': 3, 'four': 4,
                     'five': 5, 'six': 6, 'eight': 8, 'twelve': 12}}, inplace=True)


x = cars_dataSet.drop(
    {'doornumber', 'enginetype', 'fuelsystem', 'compressionratio', 'citympg', 'highwaympg', 'car_ID', 'symboling', 'carbody', 'CarName', 'wheelbase', 'carwidth', 'carlength', 'carheight', 'curbweight', 'boreratio', 'stroke', 'price'}, axis=1)

y = cars_dataSet['price']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.8, random_state=2)
# print(x_train)


model = LinearRegression()
model.fit(x_train, y_train)


prediction = model.predict(x_test)


print(prediction)

mse = metrics.r2_score(y_test, prediction)
print("mse : ", mse)

plt.scatter(y_test, prediction)
plt.ylabel("actual price")
plt.xlabel("predicted price")
plt.title('comp')
plt.show()
