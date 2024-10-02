# src/stages/split_data.py
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
import json

# Load parameters from params.yaml
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)



# Access parameters from the yaml file
raw_data_path = params['data']['raw_data_path']
processed_data_path = params['data']['processed_data_path']
test_size = params['data_split']['test_size']
random_seed = params['base']['random_seed']

# Load raw data
df = pd.read_csv(raw_data_path)

# Split the data
X = df.drop(columns=['Exited'])
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

# Save processed data
X_train.to_csv(f"{processed_data_path}/x_train.csv", index=False)
X_test.to_csv(f"{processed_data_path}/x_test.csv", index=False)
y_train.to_csv(f"{processed_data_path}/y_train.csv", index=False)
y_test.to_csv(f"{processed_data_path}/y_test.csv", index=False)

print("Data split completed successfully.")

# Save the accuracy to a JSON file
metrics = {
    "accuracy": 70
}

with open('reports/metrics.json', 'w') as f:
    json.dump(metrics, f)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Assume y_true and y_pred are your true and predicted labels
y_true = np.array([0, 4444, 1, 0])  # Example true labels
y_pred = np.array([0, 1, 0, 0])  # Example predicted labels

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

# Plot confusion matrix
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('reports/confusion_matrix.png')  # Save the plot


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample data generation
# Let's create a sample dataset
np.random.seed(0)  # For reproducibility
x = np.linspace(0, 10, 100)  # 100 points between 0 and 10
y = np.sin(x) + np.random.normal(scale=0.1, size=x.shape)  # Sin wave with noise

# Convert to DataFrame for potential further use
data = pd.DataFrame({'x': x, 'y': y})

# Save the data for DVC tracking

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(data['x'], data['y'], marker='o', linestyle='-', color='b', label='Data Points')  # Line plot
plt.title('Sample Plot of X vs Y')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.xticks(np.arange(0, 11, 1))  # Set x ticks from 0 to 10
plt.yticks(np.arange(-1.5, 2, 0.5))  # Set y ticks



# Annotating each point with its y value
for i in range(len(data)):
    plt.annotate(f'{data["y"].iloc[i]:.2f}', (data['x'].iloc[i], data['y'].iloc[i]), 
                 textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

plt.grid()
plt.legend()
plt.tight_layout()

# Save the plot as an image
plt.savefig('reports/plot.png')
plt.show()  # Display the plot (optional)

