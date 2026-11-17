"""Lab 3 - Neural network (TensorFlow/Keras) vs Decision Tree. Script version of the notebook.
Run from repo root:  python src/lab3_neural_network.py   (requires tensorflow)
"""
import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    display
except NameError:
    display = print

os.makedirs('results', exist_ok=True)
for p in ['simple_cyber_activity.csv', 'data/simple_cyber_activity.csv', '../data/simple_cyber_activity.csv']:
    if os.path.exists(p):
        DATA_PATH = p
        break
else:
    raise FileNotFoundError('simple_cyber_activity.csv not found - upload it first.')
print('Using dataset:', DATA_PATH)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(DATA_PATH)
print('Dataset shape:', df.shape)
print('Class counts (0=Normal, 1=Attack):')
print(df['label'].value_counts().sort_index())
display(df.head())

features = ['failed_logins', 'login_hour', 'new_device', 'data_transfer_mb']
X = df[features]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on training data
X_test_scaled = scaler.transform(X_test)          # reuse the same scaler

print('Training shape:', X_train_scaled.shape)
print('Testing shape :', X_test_scaled.shape)
print('Scaled training mean (~0):', X_train_scaled.mean(axis=0).round(3))
print('Scaled training std  (~1):', X_train_scaled.std(axis=0).round(3))

import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.20,
    epochs=20,
    batch_size=8,
    verbose=1
)

print(f"\nFinal training accuracy  : {history.history['accuracy'][-1] * 100:.2f}%")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")

plt.figure()
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Neural Network Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
plt.savefig('results/q4_accuracy_curve.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Neural Network Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.savefig('results/q4_loss_curve.png', dpi=150, bbox_inches='tight')
plt.close()

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')
print(f'Test loss: {test_loss:.4f}')

# --- auto-generated interpretation (adapts to YOUR run) ---
h = history.history
gap = h['accuracy'][-1] - h['val_accuracy'][-1]
print('\nInterpretation:')
print(f"* Training loss fell from {h['loss'][0]:.3f} to {h['loss'][-1]:.3f}; validation loss went from {h['val_loss'][0]:.3f} to {h['val_loss'][-1]:.3f}.")
print(f"* Final train/validation accuracy = {h['accuracy'][-1]*100:.1f}% / {h['val_accuracy'][-1]*100:.1f}% (gap {gap*100:+.1f} points).")
if gap > 0.10:
    print('* A large train>validation gap suggests possible overfitting.')
elif h['accuracy'][-1] < 0.80 and h['loss'][-1] > 0.4:
    print('* Accuracy is still modest and loss is still falling: the model is mildly UNDERFITTING after only 20 epochs; more epochs would likely help.')
else:
    print('* Training and validation performance are close: no strong sign of overfitting.')
if h['val_loss'][-1] < h['val_loss'][-4] - 0.02:
    print('* Validation loss was still falling at epoch 20, so the model had not fully converged; more epochs could help.')
if test_accuracy < h['val_accuracy'][-1] - 0.08:
    print('* Test accuracy is lower than validation accuracy; with only 15 validation and 25 test rows each estimate is noisy (one record = 4-7 points).')

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

new_records = pd.DataFrame([
    {'failed_logins': 0, 'login_hour': 11, 'new_device': 0, 'data_transfer_mb': 35},
    {'failed_logins': 7, 'login_hour': 1,  'new_device': 1, 'data_transfer_mb': 250},
    {'failed_logins': 2, 'login_hour': 23, 'new_device': 1, 'data_transfer_mb': 850}
])

new_scaled = scaler.transform(new_records)
probabilities = model.predict(new_scaled, verbose=0).flatten()
new_records['nn_probability'] = probabilities.round(3)
new_records['nn_prediction'] = np.where(probabilities >= 0.50, 'Attack', 'Normal')

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_pred)
new_records['tree_prediction'] = np.where(tree.predict(new_records[features]) == 1, 'Attack', 'Normal')
new_records['agree'] = new_records['nn_prediction'] == new_records['tree_prediction']

print(f'Neural-network accuracy: {test_accuracy * 100:.2f}%')
print(f'Decision-tree accuracy: {tree_accuracy * 100:.2f}%')
print('Tree confusion matrix [Normal, Attack]:')
print(confusion_matrix(y_test, tree_pred, labels=[0, 1]))
display(new_records)

comparison = pd.DataFrame({
    'model': ['Neural network (4-8-4-1)', 'Decision Tree (max_depth=4)'],
    'test_accuracy_%': [round(test_accuracy * 100, 2), round(tree_accuracy * 100, 2)]})
display(comparison)
comparison.to_csv('results/q5_model_comparison.csv', index=False)
new_records.to_csv('results/q5_new_record_predictions.csv', index=False)

print('\nAgreement analysis:')
for i, r in new_records.iterrows():
    status = 'AGREE (higher confidence)' if r['agree'] else 'DISAGREE -> send for human review'
    near = ' [probability near 0.50 = uncertain]' if abs(r['nn_probability'] - 0.5) < 0.15 else ''
    print(f"Record {i+1}: NN={r['nn_prediction']} ({r['nn_probability']}), Tree={r['tree_prediction']} -> {status}{near}")

