# RETAIN Model - Homework 4

This notebook implements the RETAIN (REverse Time AttentIoN) model to provide interpretable predictions on healthcare time-series data. The focus is on building and training the model using patient electronic health records (EHRs).

---

## 📁 Dataset Description

The data files are located in a path defined as `../HW4_RETAIN-lib/data/`, and consist of the following:

- **`pids.pkl`**: Patient IDs  
- **`vids.pkl`**: Visit IDs (multiple per patient)  
- **`hfs.pkl`**: Heart failure labels (1 or 0)  
- **`seqs.pkl`**: Sequences of diagnosis codes per visit  
- **`types.pkl`**: List of all diagnosis code types  
- **`rtypes.pkl`**: Reverse mapping of code labels to readable strings  

These are loaded using `pickle` for processing.

---

## 🧪 Data Exploration

The notebook explores a single patient’s data (index 3) to display:

- Patient ID  
- Heart failure label  
- Visit details  
- Diagnosis codes (raw and readable)

It also prints the total number and proportion of heart failure patients in the training data.

---

## 🧱 PyTorch Dataset Class

A custom PyTorch Dataset class is defined to handle patient data in batches during training.

```python
class CustomDataset(Dataset):
    def __init__(self, seqs, hfs):
        self.x = seqs
        self.y = hfs

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
