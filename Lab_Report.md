# Cybersecurity Lab Assignment 3 Report – Classifying Cyber Activity Using a Simple Neural Network

**Student Name:** Aditya  
**Roll Number:** 2401010245

> Numbers below come from the **reference run** (NumPy stand-in for the Keras model – see README). If your Colab/TensorFlow run prints slightly different values, use yours.

## Q1. CNN, RNN/LSTM and GAN concept matching
| Model | Suitable data | Cyber-security example |
|---|---|---|
| CNN | Images / grid-like data | Malware image classification (bytes rendered as a greyscale image); packet matrices |
| RNN / LSTM | Sequences / logs / time series | Sequential intrusion or authentication-log analysis |
| GAN | Synthetic-data generation | Generate controlled synthetic attack samples for augmentation |
| Dense network | Small numeric feature table | Normal/Attack classification in this lab |

**Why dense:** the input is only four independent numeric features – no image grid, no ordered sequence and no need to generate data – so a fully connected network is the simplest suitable architecture.

## Q2. Dataset preparation, split and normalization
* Dataset: 100 rows, **70 Normal (0)** and **30 Attack (1)**.
* Split 75 / 25 (stratified, `random_state=42`): **training (75, 4)**, **testing (25, 4)**.
* **Normalization:** `StandardScaler` rescales each feature to mean 0 and std 1 so that `data_transfer_mb` does not dominate small-valued features; it is fitted on the training data only and reused for the test set and new records.

## Q3. Network architecture and training
* 4 → 8 (ReLU) → 4 (ReLU) → 1 (sigmoid); parameters 40 + 36 + 5 = **81**. Adam, binary cross-entropy, 20 epochs, `validation_split=0.20`, `batch_size=8`.
* Final **training accuracy 90.00 %**, **validation accuracy 86.67 %**.
* **Epoch:** one full pass through the training data (60 rows after the validation split). **Batch size:** number of rows processed before one weight update (8, so 8 updates per epoch).

## Q4. Curves and test evaluation
* **Test accuracy = 76.00 %**, **test loss = 0.4664** (graphs: `results/reference_run/q4_accuracy_curve.png`, `q4_loss_curve.png`).
* **Interpretation:** training loss fell steadily (0.660 → 0.397) and validation loss followed (0.705 → 0.498), while training and validation accuracy are close (90.0 % / 86.7 %), so there is no strong sign of overfitting. Validation loss was still decreasing at epoch 20, i.e. the model is slightly under-trained and more epochs would probably help. The lower test accuracy (76 %) than validation accuracy is partly sampling noise: with 15 validation and 25 test rows, one record changes accuracy by 4–7 points.

## Q5. New predictions and Decision Tree comparison
| failed_logins | login_hour | new_device | data_transfer_mb | NN probability | NN prediction | Tree prediction |
|---|---|---|---|---|---|---|
| 0 | 11 | 0 | 35 | 0.356 | Normal | Normal |
| 7 | 1 | 1 | 250 | 0.837 | Attack | Attack |
| 2 | 23 | 1 | 850 | 0.786 | Attack | Attack |

| Model | Test accuracy |
|---|---|
| Neural network (4-8-4-1) | **76.00 %** |
| Decision Tree (max_depth=4) | **88.00 %** (confusion matrix `[[16, 2], [1, 6]]`) |

* **Agreement analysis:** both models agree on all three records, which raises confidence. Record 1 is a clear Normal pattern (no failures, office hours, known device, small transfer) although the neural probability 0.356 is not far from the 0.50 threshold. Record 2 (7 failed logins, 01:00, new device) is a brute-force-like pattern, and record 3 (new device, 23:00, 850 MB) looks like data exfiltration. Had the models disagreed, the record should go to a human analyst.
* **Why the simpler model may be chosen:** on this small table the Decision Tree is more accurate (88 % vs 76 %), trains instantly, needs no scaling or TensorFlow, and its if-then rules can be read, audited and explained to a security analyst.
* **Conclusion:** for a small four-feature table the explainable Decision Tree is the better choice here; neural networks pay off with far more data or raw, high-dimensional inputs (images, sequences).
