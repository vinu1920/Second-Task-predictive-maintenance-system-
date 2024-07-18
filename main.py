import csv
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# Example data
data = {
    'equipment_id': [1122, 1123, 1112],
    'timestamp': [30, 25, 31],
    'sensor_reading': [1, 2, 3],
    'failure': [1, 1, 0]
}

df = pd.DataFrame(data)

# Features and target
X = df[['equipment_id', 'timestamp', 'sensor_reading']]
y = df['failure']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train a simple model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Calculate precision and recall with zero_division parameter
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
print(f'Precision: {precision}')
print(f'Recall: {recall}')

# Classification report
report = classification_report(y_test, y_pred, zero_division=1)
print(report)

# Data to be written to the CSV file
data = [
    ["equipment_id", "timestamp", "sensor_reading", "failure"],
    ["1122", 30, 1, 1],
    ["1123", 25, 2, 1],
    ["1112", 31, 3, 0]
]

# Open a file for writing
with open('equipment_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write each row of data to the CSV file
    for row in data:
        writer.writerow(row)

# Load the dataset
data = pd.read_csv('equipment_data.csv')

# Preview the dataset
print(data.head())

# Fill missing values (if any)
data = data.ffill()

# Normalize sensor data
scaler = StandardScaler()
sensor_columns = ['sensor_reading']
data[sensor_columns] = scaler.fit_transform(data[sensor_columns])

# Example: Rolling average of sensor readings
for sensor in sensor_columns:
    data[f'{sensor}_rolling_mean'] = data[sensor].rolling(window=2).mean()

# Drop rows with NaN values created by rolling mean
data.dropna(inplace=True)

# Define features and target
X = data.drop(['equipment_id', 'timestamp', 'failure'], axis=1)
y = data['failure']

# Check if the dataset is sufficient for splitting
if len(data) > 1:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(model, 'predictive_maintenance_model.pkl')

    # Load the model (for prediction)
    model = joblib.load('predictive_maintenance_model.pkl')
else:
    print("Not enough data to split into training and testing sets.")
