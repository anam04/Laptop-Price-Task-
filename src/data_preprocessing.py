import pandas as pd
import numpy as npgi
import matplotlib.pyplot as plt #math operations

# Load data
df = pd.read_csv("laptop_price_task/data/Laptop Price.csv")
# ---------------- Basic Checks ----------------
print("Shape:", df.shape)
print("Duplicates:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Nulls:\n", df.isnull().sum())
# ---------------- Clean RAM & Weight ----------------
df["Ram"] = df["Ram"].astype(str).str.replace("GB", "", regex=False).astype(int)
df["Weight"] = df["Weight"].astype(str).str.replace("kg", "", regex=False).astype(float)
# ---------------- Screen Resolution Features ----------------
sr = df["ScreenResolution"].astype(str)
# Flags
df["Touchscreen"] = sr.str.contains("Touch", case=False, na=False).astype(int)
df["Ips"] = sr.str.contains("IPS", case=False, na=False).astype(int)
# Extract X_res and Y_res with regex
xy = sr.str.extract(r'(\d+)\s*x\s*(\d+)', expand=True)
df["X_res"] = pd.to_numeric(xy[0], errors="coerce")
df["Y_res"] = pd.to_numeric(xy[1], errors="coerce")
# Ensure Inches is numeric
df["Inches"] = pd.to_numeric(df["Inches"], errors="coerce")
# Compute PPI
with np.errstate(divide="ignore", invalid="ignore"):
    df["ppi"] = np.sqrt(df["X_res"]**2 + df["Y_res"]**2) / df["Inches"]
df["ppi"] = df["ppi"].replace([np.inf, -np.inf], np.nan).astype(float)
# ---------------- CPU Brand ----------------
df["Cpu name"] = df["Cpu"].apply(lambda x: " ".join(x.split()[0:3]))
def fetch_processor(text):
    if text in ["Intel Core i7", "Intel Core i5", "Intel Core i3"]:
        return text
    elif text.split()[0] == "Intel":
        return "Other Intel Processor"
    else:
        return "AMD Processor"

df["Processor brand"] = df["Cpu name"].apply(fetch_processor)

# ---------------- Memory Features ----------------
mem = df["Memory"].astype(str).str.upper()
mem = mem.str.replace(".0", "", regex=False)
mem = mem.str.replace("GB", "", regex=False)
mem = mem.str.replace("TB", "000", regex=False)

# Split into 2 parts
parts = mem.str.split("+", n=1, expand=True)
df["first"] = parts[0].str.strip()
df["second"] = parts[1].fillna("0").str.strip()

# Flags
df["Layer1HDD"] = df["first"].str.contains("HDD", na=False).astype(int)
df["Layer1SSD"] = df["first"].str.contains("SSD", na=False).astype(int)
df["Layer1Hybrid"] = df["first"].str.contains("HYBRID", na=False).astype(int)
df["Layer1Flash_Storage"] = df["first"].str.contains("FLASH STORAGE", na=False).astype(int)

df["Layer2HDD"] = df["second"].str.contains("HDD", na=False).astype(int)
df["Layer2SSD"] = df["second"].str.contains("SSD", na=False).astype(int)
df["Layer2Hybrid"] = df["second"].str.contains("HYBRID", na=False).astype(int)
df["Layer2Flash_Storage"] = df["second"].str.contains("FLASH STORAGE", na=False).astype(int)

# Keep only digits
df["first"] = pd.to_numeric(df["first"].str.replace(r"\D", "", regex=True), errors="coerce").fillna(0).astype(int)
df["second"] = pd.to_numeric(df["second"].str.replace(r"\D", "", regex=True), errors="coerce").fillna(0).astype(int)

# Final storage features
df["HDD"] = df["first"]*df["Layer1HDD"] + df["second"]*df["Layer2HDD"]
df["SSD"] = df["first"]*df["Layer1SSD"] + df["second"]*df["Layer2SSD"]
df["Hybrid"] = df["first"]*df["Layer1Hybrid"] + df["second"]*df["Layer2Hybrid"]
df["Flash_Storage"] = df["first"]*df["Layer1Flash_Storage"] + df["second"]*df["Layer2Flash_Storage"]

# Drop helper columns
df.drop(columns=["Memory","first","second",
                 "Layer1HDD","Layer1SSD","Layer1Hybrid","Layer1Flash_Storage",
                 "Layer2HDD","Layer2SSD","Layer2Hybrid","Layer2Flash_Storage"],
        inplace=True, errors="ignore")

# ---------------- GPU Brand ----------------
df["Gpu Brand"] = df["Gpu"].apply(lambda x: x.split()[0])
df = df[df["Gpu Brand"] != "ARM"]  # drop rare ARM GPUs

# ---------------- Operating System ----------------
def cat_os(inp):
    if inp in ["Windows 10", "Windows 7", "Windows 10 S"]:
        return "Windows"
    elif inp in ["macOS", "Mac OS X"]:
        return "Mac"
    else:
        return "Others/No OS/Linux"

df["os"] = df["OpSys"].apply(cat_os)

# ---------------- Final Check ----------------
print(df.head(10))
print(df.select_dtypes(include=[np.number]).corr()["Price"].sort_values(ascending=False))