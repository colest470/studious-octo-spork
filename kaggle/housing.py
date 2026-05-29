import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

iowa_file_path = '../input/home-data-for-ml-course/train.csv'

housing_data = pd.read_csv(iowa_file_path)

housing_data.describe()

X = housing_data[[['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]]

y = housing_data.Price

model = DecisionTreeRegressor(random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
MAE = mean_absolute_error(y_test, predictions)

print(f"MAE error: {MAE:.4f}")