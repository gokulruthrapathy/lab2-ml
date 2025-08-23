

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Utility functions ----------
def load_csv(path):
    """Load CSV into DataFrame"""
    return pd.read_csv(path)

def basic_summary(df):
    """Return shape, dtypes, head, and describe"""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes,
        "head": df.head(),
        "describe": df.describe()
    }

def mark_high_sum(df, cols, threshold):
    """Mark rows with sum of selected columns > threshold"""
    df2 = df.copy()
    df2["HIGH_SUM"] = np.where(df2[cols].sum(axis=1) > threshold, "HIGH", "LOW")
    return df2

def plot_channel(df, channel_name):
    """Plot a numeric column over index"""
    plt.figure(figsize=(10,5))
    plt.plot(df[channel_name])
    plt.title(f"Channel {channel_name} over Index")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

def jaccard_binary(a, b):
    """Jaccard similarity for binary arrays"""
    a, b = np.array(a), np.array(b)
    f11 = np.sum((a==1) & (b==1))
    f10 = np.sum((a==1) & (b==0))
    f01 = np.sum((a==0) & (b==1))
    denom = f11 + f10 + f01
    return f11 / denom if denom != 0 else 0

def smc_binary(a, b):
    """Simple Matching Coefficient for binary arrays"""
    a, b = np.array(a), np.array(b)
    f11 = np.sum((a==1) & (b==1))
    f00 = np.sum((a==0) & (b==0))
    return (f11 + f00) / len(a)

def compute_similarity_metrics(df, row1, row2, bin_cols=10):
    """Compute JC, SMC, COS between two rows"""
    # binary conversion for JC and SMC
    Xbin = (df.iloc[[row1,row2], :bin_cols] > df.iloc[:, :bin_cols].median()).astype(int)
    jc = jaccard_binary(Xbin.iloc[0], Xbin.iloc[1])
    smc = smc_binary(Xbin.iloc[0], Xbin.iloc[1])
    cos = cosine_similarity([df.iloc[row1]], [df.iloc[row2]])[0,0]
    return jc, smc, cos

def similarity_heatmap(df, metric="cosine", top_n=20):
    """Heatmap for first top_n rows"""
    X = df.iloc[:top_n].copy()
    if metric in ["jaccard", "smc"]:
        Xbin = (X > X.median()).astype(int)
        sim = np.zeros((top_n, top_n))
        for i in range(top_n):
            for j in range(top_n):
                if metric == "jaccard":
                    sim[i,j] = jaccard_binary(Xbin.iloc[i], Xbin.iloc[j])
                else:
                    sim[i,j] = smc_binary(Xbin.iloc[i], Xbin.iloc[j])
    else:
        sim = cosine_similarity(X)
    plt.figure(figsize=(8,6))
    sns.heatmap(sim, cmap="viridis")
    plt.title(f"{metric.upper()} Similarity Heatmap (first {top_n} rows)")
    plt.show()

def impute_missing(df):
    """Impute numeric cols with mean"""
    df2 = df.copy()
    num_cols = df2.select_dtypes(include=[np.number]).columns
    imp = SimpleImputer(strategy="mean")
    df2[num_cols] = imp.fit_transform(df2[num_cols])
    return df2

def scale_features(df, method="standard"):
    """Scale numeric columns"""
    df2 = df.copy()
    num_cols = df2.select_dtypes(include=[np.number]).columns
    if method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    df2[num_cols] = scaler.fit_transform(df2[num_cols])
    return df2

# ---------- Main program ----------
if __name__ == "__main__":
    # A1: Load data
    df = load_csv("features_raw.csv")
    
    # A2: Mark HIGH_SUM
    df_marked = mark_high_sum(df, cols=df.columns[:5], threshold=1000)
    
    # A3: Plot one channel
    plot_channel(df, channel_name=df.columns[0])
    
    # A4: Summary stats
    summary = basic_summary(df)
    
    # A5/A6: Similarity between first two rows
    jc, smc, cos = compute_similarity_metrics(df, 0, 1)
    
    # A7: Heatmaps
    similarity_heatmap(df, metric="jaccard")
    similarity_heatmap(df, metric="smc")
    similarity_heatmap(df, metric="cosine")
    
    # A8: Imputation
    df_imputed = impute_missing(df)
    
    # A9: Scaling
    df_scaled = scale_features(df_imputed, method="standard")
    
    # Print outputs
    print("A1 Shape:", summary["shape"])
    print("A2 High sum counts:\n", df_marked["HIGH_SUM"].value_counts())
    print("A4 Dtypes:\n", summary["dtypes"])
    print("A5/A6 Similarities -> JC:", jc, "SMC:", smc, "COS:", cos)
    print("A8 Any missing after impute?:", df_imputed.isnull().sum().sum())
    print("A9 First row after scaling:", df_scaled.iloc[0, :5])

