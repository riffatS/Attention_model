# Attention_model
# RETAIN Model - Homework 4

This Jupyter notebook demonstrates the implementation and data preparation pipeline for the **RETAIN (REverse Time AttentIoN)** model. The RETAIN model is a neural architecture designed to provide interpretable predictions on electronic health records using a reversed recurrent attention mechanism.

---

## 📁 Dataset

The dataset used for training the model is stored in a directory specified by `DATA_PATH`. It includes preprocessed EHR (Electronic Health Record) data for multiple patients, which consists of:

- `pids.pkl`: Patient IDs
- `vids.pkl`: Visit IDs (each patient has multiple visits)
- `hfs.pkl`: Heart failure label (1 if the patient had heart failure, else 0)
- `seqs.pkl`: Diagnosis codes per visit
- `types.pkl`: List of unique medical codes
- `rtypes.pkl`: Reverse lookup of diagnosis codes to readable names

These files are loaded using `pickle`:

```python
pids = pickle.load(open(os.path.join(DATA_PATH,'train/pids.pkl'), 'rb'))
