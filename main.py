import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Optimized function to load CSV files with reduced memory usage
def load_csv(file_path, use_columns=None):
    data = pd.read_csv(file_path, usecols=use_columns, parse_dates=['date'], low_memory=False)
    return data

# Load datasets with only the necessary columns
train_columns = ['date', 'store_nbr', 'sales']  # Add necessary columns
test_columns = ['date', 'store_nbr']  # Add necessary columns
oil_columns = ['date', 'dcoilwtico']
holidays_columns = ['date', 'type']
stores_columns = ['store_nbr', 'city', 'state', 'type']

train = load_csv('/kaggle/input/store-sales-time-series-forecasting/train.csv', use_columns=train_columns)
test = load_csv('/kaggle/input/store-sales-time-series-forecasting/test.csv', use_columns=test_columns)
oil = load_csv('/kaggle/input/store-sales-time-series-forecasting/oil.csv', use_columns=oil_columns)
holidays_events = load_csv('/kaggle/input/store-sales-time-series-forecasting/holidays_events.csv', use_columns=holidays_columns)
stores = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/stores.csv', usecols=stores_columns)

# Merge datasets dynamically
train = train.merge(oil[['date', 'dcoilwtico']], on='date', how='left')
test = test.merge(oil[['date', 'dcoilwtico']], on='date', how='left')
train = train.merge(holidays_events[['date', 'type']], on='date', how='left')
test = test.merge(holidays_events[['date', 'type']], on='date', how='left')
train = train.merge(stores[['store_nbr', 'city', 'state', 'type']], on='store_nbr', how='left')
test = test.merge(stores[['store_nbr', 'city', 'state', 'type']], on='store_nbr', how='left')

# Preprocessing
def preprocess(data, is_train=True):
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['day_of_week'] = data['date'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

    data['dcoilwtico'] = data['dcoilwtico'].ffill().bfill()

    if is_train and 'transactions' in data.columns:
        data['transactions'] = data['transactions'].fillna(0)
    elif 'transactions' in data.columns:
        data['transactions'] = data['transactions'].fillna(0)

    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']

    le = LabelEncoder()
    for col in categorical_columns:
        data[col] = le.fit_transform(data[col])

    columns_to_drop = ['date', 'locale', 'locale_name', 'description', 'transferred']
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    data = data.drop(columns_to_drop, axis=1, errors='ignore')

    return data

train_processed = preprocess(train, is_train=True)
test_processed = preprocess(test, is_train=False)

X = train_processed.drop('sales', axis=1)
y = train_processed['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use fewer estimators for quicker results
model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)  # Limit tree depth
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Predict on the test data for Kaggle submission
test_predictions = model.predict(test_processed.drop('id', axis=1, errors='ignore'))
submission = pd.DataFrame({'id': test['id'], 'sales': test_predictions})
submission.to_csv('submission.csv', index=False)
