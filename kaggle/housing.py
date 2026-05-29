import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

iowa_file_path = '../input/home-data-for-ml-course/train.csv'

housing_data = pd.read_csv(iowa_file_path)

housing_data.describe()

X = housing_data[[['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]]

y = housing_data.Price

def get_mae(max_leaf_nodes, train_X, X_test, train_y, y_test):
    model = DecisionTreeRegressor(random_state=0, max_leaf_nodes=max_leaf_nodes)
    model.fit(train_X, train_y)
    predicted = model.predict(X_test)
    mae = mean_absolute_error(y_test, predicted)

    return mae

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

mae_arr = []

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=1)

for max_leaf_nodes in candidate_max_leaf_nodes:
    mae_arr.append(get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test))
    
position = 0
least = 0
for index, mae in enumerate(mae_arr):
    if index == 0:
        least = mae_arr[index]

    if least > mae:
        least = mae
        position = index
        

best_tree_size = candidate_max_leaf_nodes[position]

model = DecisionTreeRegressor(random_state=0, max_leaf_nodes=best_tree_size)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
MAE = mean_absolute_error(y_test, predictions)

print(f"MAE error: {MAE:.4f}")
