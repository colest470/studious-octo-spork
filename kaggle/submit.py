import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

data_path_train = "/home/mark/Documents/convolutionalJulia/data/train.csv"
data_path_test = "/home/mark/Documents/convolutionalJulia/data/test.csv"

housing_data_train = pd.read_csv(data_path_train)
housing_data_test = pd.read_csv(data_path_test)

print(housing_data_test.columns[len(housing_data_test.columns) - 1])

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X_train = housing_data_train[features]
y_train = housing_data_train["SalePrice"]
X_test = housing_data_test[features]

model = RandomForestRegressor(random_state=1)

model.fit(X_train, y_train)

predicted = model.predict(X_test)

# mae = mean_absolute_error(, predicted)

output = pd.DataFrame({'Id': housing_data_test.Id,
                       'SalePrice': predicted})
output.to_csv('submission.csv', index=False)