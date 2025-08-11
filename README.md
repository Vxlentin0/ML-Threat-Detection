# ML Threat-Detection (TensorFlow & scikit-learn)
A compact, end-to-end machine learning pipeline for binary threat detection on tabular data. The project trains a neural network with TensorFlow/Keras: handles class imbalance, tunes a decision threshold on a validation set, and saves reusable artifacts (model + threshold + metrics). Optionally, you can enable TensorBoard logging and add a fast scikit-learn baseline.


🔍 What It Does
=================

Trains a machine learning model (a TensorFlow/Keras MLP) to classify records as `threat` vs `non-threat` 

This program also accomplishes: 

1. Preventing data leaks by learning normalization statistics **only on the training set**

2. Handles class imbalance via class weights

3. Uses **early stopping** and **model checkpointing** to avoid overfitting and keep the best model.

4. Tunes a **classification threshold** on the validation set (e.g., maximising F1) and evaluates on a held-out test set.

5. Saves reusable artifafts :

    - `models/best_model.keras` - the trained model **including preprocessing**
    - `models/threshold.json` - the chosen decision cutoff for inference
    - `outputs/*` - human-readable & machine-readable evaluation results

This pipeline is ML-ready right out of the box  and can also be further extended with scikit-learn baselines or even richer preprocessing for categorical/text features.


⚙️ How It Works
===================

## Data → Splits → Model → Train → Tune Threshold → Evaluate → Save Artifacts 
 
### Load & Clean (`load_dataframe`)

Reads `data/threats_dataset.csv`, separates features/label (`is_threat`), performes light numerical cleanup (median imputation) and simple encoding fallbacks for non-numerical columns. <br/>


### Leakage-safe Splits (`make_splits`) 

Stratified Train/Val/Test splits to presewrve label ratios. Normalization stats are learned **only** from the training data. <br/>


### Model with In-Graph Normalization (`build_model`) 

A small MLP with a `tf.keras.layers.Normalization` layer. The normalization is part of the saved model, so you don't need a separate scaler at inference. <br/>


### Training 

Uses class weights for imbalance, `EarlyStopping(monitor=val_auc)` and `ModelCheckpoint` to save `models/best_model.keras`. Optional TensorBoard logging can be enabled. <br/>


### Threshold Tuning (`pick_threshold`)

Chooses the decision threshold on the **validation set** (default: maximize F1). This threshold is saved to `models/threshold.json`. <br/> 


### Evaluation & Reporting 

Reports ROC-AUC, PR-AUC, confusion matrix, classification report on the **test set** as the tuned threshold. Artifacts are written into `outputs/`. <br/>


### inference (`detect_threat`)

For new samples, load `best_model.keras` and `threshold.json`, compute probability, and classify using the stored threshold


▶️ How to Run
===============

### Prerequisites

- Python 3.9
- install dependancies: <br/> 
  ```
  pip install -r requirements.txt
  ```

### Train & Evaluate 

From the project tool: <br/> 

```
python scripts/threat_detection.py
```

Input data is expected at `data/threats_dataset.csv` with a binary label columned named `is_threat`. 

After training, you should have:

  - `models/best_model.keras`
  - `models/threshold.json`
  - `outputs/metrics.json`, `outputs/evaluation_results.txt`, `outputs/confusion_matrix.png`

*Optional (TensorBoard): uncomment the TensorBoard callback in `threat_detection.py`, then run:* 

```
tensorboard --logdir outputs/logs
```

Navigate to the shown URL to inspect training curves.


### Use the Trained Model (Example) 

In a Python shell/notebook: 

```
import json, numpy  as np, tensorflow as tf
from pathlib import Path 

model = tf.keras.models.load_model(Path('models/best_model.keras'))
thr = json.loads(Path('models/threshold.json').read_text())['threshold']

# Replace with real feature values in the same column order as training
x = np.array([[0.12, 3.4, 8.9, 1.0, 0.0, 42.0]], dtype = 'float32')
prob = float(model.predict(x, verbose = 0).ravel()[0])
print({'prob':prob, 'is_threat': prob >= thr, 'threshold': thr})
```


🔨 Notes & Extensibility
===============

- To add a scikit-learn baseline (e.g., Logistic Regression) or richer categorical/text handling, see the commented sections in `scripts/threat_detection.py`
- Pin package versions in `reqiurements.txt` to keep results consistent across machines
- For production, consider input schema validation, model versioning, and monitoring (drift, alert rate, latency).
