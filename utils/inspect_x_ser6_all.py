import numpy as np
import scipy.io as sio
import pandas as pd

mat = sio.loadmat("DATA/Exp3.mat", squeeze_me=True, struct_as_record=False)
Exp3 = mat["Exp3"]
X_ser6_all = Exp3.X_ser6_all

X = np.array(Exp3.X_ser6_all)
print("shape:", X.shape)
print("dtype:", X.dtype)

X = X_ser6_all   

df = pd.DataFrame(X, columns=["c1","c2","c3","c4","c5","c6","c7"])

for col in df.columns:
    vals = df[col].values
    nunq = np.unique(vals).size
    print(f"{col}: min={vals.min():.4g} max={vals.max():.4g} unique={nunq}")

print("\nColumn 4 unique values:", np.unique(df["c4"]))
print("Column 5 unique values:", np.unique(df["c5"]))

print("\nColumn 6 value counts:")
print(df["c6"].value_counts().sort_index())

print("\nFirst 10 rows:")
print(df.head(10))