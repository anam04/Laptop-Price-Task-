import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/Laptop Price.csv")
df.drop_duplicates(inplace=True) 

print("Shape:", df.shape)
print(df.info()) #returns consise summary of the dataframe
print(df.describe()) #returns statistical summary of the numerical columns
print("Duplicates:", df.duplicated().sum()) #checks the number of duplicate rows
df.drop_duplicates(inplace=True) #drop the duplicate rows
print("After dropping duplicates, Shape:", df.shape) #after dropping duplicates, check the shape
print("Nulls:\n", df.isnull().sum()) #Check for null values in eahc column if any 
print(df.dropna()) #drop the null values if any, in this project, there are no null values.
# Clean RAM & Weight
df["Ram"] = df["Ram"].astype(str).str.replace("GB", "", regex=False).astype(int)
df["Weight"] = df["Weight"].astype(str).str.replace("kg", "", regex=False).astype(float)

# Screen features
sr = df["ScreenResolution"].astype(str)
df["TouchScreen"] = sr.str.contains("Touch", case=False, na=False).astype(int)
df["IPS"] = sr.str.contains("IPS", case=False, na=False).astype(int)

# Resolution → X_res, Y_res
xy = sr.str.extract(r'(\d+)\s*x\s*(\d+)', expand=True)
df["X_res"] = pd.to_numeric(xy[0], errors="coerce")
df["Y_res"] = pd.to_numeric(xy[1], errors="coerce")

# Inches numeric
df["Inches"] = pd.to_numeric(df["Inches"], errors="coerce")

# PPI
with np.errstate(divide="ignore", invalid="ignore"):
    df["ppi"] = np.sqrt(df["X_res"]**2 + df["Y_res"]**2) / df["Inches"]
df["ppi"] = df["ppi"].replace([np.inf, -np.inf], np.nan).astype(float)

# Drop raw resolution cols
df.drop(columns=['ScreenResolution','X_res','Y_res','Inches'], inplace=True)

# CPU brand
def fetch_processor(text):
    text = " ".join(text.split()[0:3])
    if text in ["Intel Core i7", "Intel Core i5", "Intel Core i3"]:
        return text
    elif text.split()[0] == "Intel":
        return "Other Intel Processor"
    else:
        return "AMD Processor"

df["Cpu_brand"] = df["Cpu"].apply(fetch_processor)
df.drop(columns=["Cpu"], inplace=True)

# Memory features
#First, make everything uppercase to avoid case mismatches (like 'gb' vs 'GB')
mem = df["Memory"].astype(str).str.upper()
mem = mem.str.replace(".0", "", regex=False) # remove any trailing .0
mem = mem.str.replace("GB", "", regex=False) #remobe 'GB
mem = mem.str.replace("TB", "000", regex=False) #convert TB into 1000GB

# Sometimes laptops have 2 types of storage (like "128GB SSD + 1TB HDD"),
# so let's split them into 'first' and 'second' parts
parts = mem.str.split("+", n=1, expand=True)
df["first"] = parts[0].str.strip()
df["second"] = parts[1].fillna("0").str.strip()

# Create flags for storage types in the FIRST part
df["Layer1HDD"] = df["first"].str.contains("HDD", na=False).astype(int)
df["Layer1SSD"] = df["first"].str.contains("SSD", na=False).astype(int)
df["Layer1Hybrid"] = df["first"].str.contains("HYBRID", na=False).astype(int)
df["Layer1Flash"] = df["first"].str.contains("FLASH STORAGE", na=False).astype(int)


# Do the same thing for the SECOND part (if it exists)
df["Layer2HDD"] = df["second"].str.contains("HDD", na=False).astype(int)
df["Layer2SSD"] = df["second"].str.contains("SSD", na=False).astype(int)
df["Layer2Hybrid"] = df["second"].str.contains("HYBRID", na=False).astype(int)
df["Layer2Flash"] = df["second"].str.contains("FLASH STORAGE", na=False).astype(int)

# Numeric extraction
df["first"] = pd.to_numeric(df["first"].str.replace(r"\D", "", regex=True), errors="coerce").fillna(0).astype(int)
df["second"] = pd.to_numeric(df["second"].str.replace(r"\D", "", regex=True), errors="coerce").fillna(0).astype(int)

# Final storage
# calculate how much of each type of storage exists in total
df["HDD"] = df["first"]*df["Layer1HDD"] + df["second"]*df["Layer2HDD"]
df["SSD"] = df["first"]*df["Layer1SSD"] + df["second"]*df["Layer2SSD"]
df["Hybrid"] = df["first"]*df["Layer1Hybrid"] + df["second"]*df["Layer2Hybrid"]
df["Flash_Storage"] = df["first"]*df["Layer1Flash"] + df["second"]*df["Layer2Flash"]


# Drop the temporary helper columns since we don’t need them anymore
df.drop(columns=["Memory","first","second",
                 "Layer1HDD","Layer1SSD","Layer1Hybrid","Layer1Flash",
                 "Layer2HDD","Layer2SSD","Layer2Hybrid","Layer2Flash"],
        inplace=True, errors="ignore")

# GPU brand
# GPUs are written like "Nvidia GeForce GTX 1050" , wqe just want the brand ("Nvidia")
df["Gpu_brand"] = df["Gpu"].apply(lambda x: x.split()[0])
# Drop the one rare case of "ARM" GPUs (they're almost nonexistent, just noise)
#checked via EDA in data exploaration notebook
df = df[df["Gpu_brand"] != "ARM"]
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

print(df.head())


# Save final cleaned dataset
df.to_csv("data/train_data.csv", index=False)
print(df.columns) 
print("Cleaned dataset saved as data/train_data.csv")
