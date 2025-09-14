import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("data/Laptop Price.csv")
df.drop_duplicates(inplace=True)
print("Initial shape:", df.shape)
print("Nulls:\n", df.isnull().sum())
# Clean RAM & Weight
df["Ram"] = df["Ram"].astype(str).str.replace("GB", "", regex=False).astype(int)
df["Weight"] = df["Weight"].astype(str).str.replace("kg", "", regex=False).astype(float)

# Screen features
sr = df["ScreenResolution"].astype(str)
df["TouchScreen"] = sr.str.contains("Touch", case=False, na=False).astype(int)
df["IPS"] = sr.str.contains("IPS", case=False, na=False).astype(int)

# Extract resolution
xy = sr.str.extract(r'(\d+)\s*x\s*(\d+)', expand=True)
df["X_res"] = pd.to_numeric(xy[0], errors="coerce")
df["Y_res"] = pd.to_numeric(xy[1], errors="coerce")

# Aspect ratio
df["aspect_ratio"] = df["X_res"] / df["Y_res"]
# Inches numeric
df["Inches"] = pd.to_numeric(df["Inches"], errors="coerce")
# PPI
with np.errstate(divide="ignore", invalid="ignore"):
    df["ppi"] = np.sqrt(df["X_res"]**2 + df["Y_res"]**2) / df["Inches"]
df["ppi"] = df["ppi"].replace([np.inf, -np.inf], np.nan).astype(float)

# Drop raw resolution cols
df.drop(columns=['ScreenResolution','X_res','Y_res','Inches'], inplace=True)

# CPU brand extraction
def fetch_processor(text):
    text = " ".join(text.split()[0:3])
    if text in ["Intel Core i9", "Intel Core i7", "Intel Core i5", "Intel Core i3"]:
        return text
    elif text.split()[0] == "Intel":
        return "Other Intel Processor"
    else:
        return "AMD/Other Processor"

df["Cpu_brand"] = df["Cpu"].apply(fetch_processor)
df.drop(columns=["Cpu"], inplace=True)


# Memory features
mem = df["Memory"].astype(str).str.upper()
mem = mem.str.replace(".0", "", regex=False)
mem = mem.str.replace("GB", "", regex=False)
mem = mem.str.replace("TB", "1024", regex=False)  # 1TB = 1024GB

parts = mem.str.split("+", n=1, expand=True)
df["first"] = parts[0].str.strip()
df["second"] = parts[1].fillna("0").str.strip()

# Storage type flags
def storage_flags(series, keyword):
    return series.str.contains(keyword, na=False).astype(int)

df["Layer1HDD"] = storage_flags(df["first"], "HDD")
df["Layer1SSD"] = storage_flags(df["first"], "SSD")
df["Layer1Hybrid"] = storage_flags(df["first"], "HYBRID")
df["Layer1Flash"] = storage_flags(df["first"], "FLASH STORAGE")

df["Layer2HDD"] = storage_flags(df["second"], "HDD")
df["Layer2SSD"] = storage_flags(df["second"], "SSD")
df["Layer2Hybrid"] = storage_flags(df["second"], "HYBRID")
df["Layer2Flash"] = storage_flags(df["second"], "FLASH STORAGE")

# Numeric extraction
df["first"] = pd.to_numeric(df["first"].str.replace(r"\D", "", regex=True), errors="coerce").fillna(0).astype(int)
df["second"] = pd.to_numeric(df["second"].str.replace(r"\D", "", regex=True), errors="coerce").fillna(0).astype(int)

# Final storage
df["HDD"] = df["first"]*df["Layer1HDD"] + df["second"]*df["Layer2HDD"]
df["SSD"] = df["first"]*df["Layer1SSD"] + df["second"]*df["Layer2SSD"]
df["Hybrid"] = df["first"]*df["Layer1Hybrid"] + df["second"]*df["Layer2Hybrid"]
df["Flash_Storage"] = df["first"]*df["Layer1Flash"] + df["second"]*df["Layer2Flash"]

# Drop helpers
df.drop(columns=["Memory","first","second",
                 "Layer1HDD","Layer1SSD","Layer1Hybrid","Layer1Flash",
                 "Layer2HDD","Layer2SSD","Layer2Hybrid","Layer2Flash"],
        inplace=True, errors="ignore")


# GPU brand
df["Gpu_brand"] = df["Gpu"].apply(lambda x: x.split()[0])
df = df[df["Gpu_brand"] != "ARM"]  # remove rare GPU
df.drop(columns=["Gpu"], inplace=True)


# OS grouping
def cat_os(inp):
    inp = str(inp)
    if "Windows" in inp:
        return "Windows"
    elif "Mac" in inp:
        return "Mac"
    elif "Linux" in inp:
        return "Linux"
    else:
        return "Other"

df["os"] = df["OpSys"].apply(cat_os)
df.drop(columns=["OpSys"], inplace=True)


# Encode categoricals

categorical_cols = ["Company", "TypeName", "os", "Gpu_brand", "Cpu_brand"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# Target transformation

import seaborn as sns
sns.histplot(df["Price"], kde=True)
plt.title("Raw Price Distribution")
plt.show()

sns.histplot(np.log1p(df["Price"]), kde=True)
plt.title("Log-Transformed Price Distribution")
plt.show()

df["Price_log"] = np.log1p(df["Price"])


# Save final cleaned dataset
df.to_csv("data/train_data.csv", index=False)
print("Final shape:", df.shape)
print("Columns:\n", df.columns)
<<<<<<< HEAD
print("Cleaned dataset saved as data/train_data.csv")
=======
print("Cleaned dataset saved as data/train_data.csv")
>>>>>>> 8dc88adde8711fd9a3672481aec1d578089a634e
