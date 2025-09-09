import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data\Laptop Price.csv")  
print(df.head())  # preview first rows
#Basic Checks
print("Shape:", df.shape) #dimension of the dataframe
print(df.info()) #returns consise summary of the dataframe
print(df.describe()) #returns statistical summary of the numerical columns
print("Duplicates:", df.duplicated().sum()) #checks the number of duplicate rows
df.drop_duplicates(inplace=True) #drop the duplicate rows
print("After dropping duplicates, Shape:", df.shape) #after dropping duplicates, check the shape
print("Nulls:\n", df.isnull().sum()) #Check for null values in eahc column if any
print(df.dropna()) #drop the null values if any, in this project, there are no null values.

#Clean RAM and Weight columns
df["Ram"] = df["Ram"].astype(str).str.replace("GB", "", regex=False).astype(int)
df["Weight"] = df["Weight"].astype(str).str.replace("kg", "", regex=False).astype(float)

#Seperate ScreenResolution into multiple features
sr = df["ScreenResolution"].astype(str)
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

# CPU_Brand Feature
df["Cpu name"] = df["Cpu"].apply(lambda x: " ".join(x.split()[0:3]))
def fetch_processor(text):
    if text in ["Intel Core i7", "Intel Core i5", "Intel Core i3"]:
        return text
    elif text.split()[0] == "Intel":
        return "Other Intel Processor"
    else:
        return "AMD Processor"

df["Processor brand"] = df["Cpu name"].apply(fetch_processor)
df["Cpu"].value_counts() #checks the count of unique value in the CPU column







#Memory Features
mem = df["Memory"].astype(str).str.upper()
mem = mem.str.replace(".0", "", regex=False)
mem = mem.str.replace("GB", "", regex=False)
mem = mem.str.replace("TB", "000", regex=False)
df['Memory'].value_counts()


# Split into 2 parts
parts = mem.str.split("+", n=1, expand=True)
df["first"] = parts[0].str.strip()
df["second"] = parts[1].fillna("0").str.strip()

# 
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

# Drop irrelevant columns, as figured from EDA
df.drop(columns=["Memory","first","second",
                 "Layer1HDD","Layer1SSD","Layer1Hybrid","Layer1Flash_Storage",
                 "Layer2HDD","Layer2SSD","Layer2Hybrid","Layer2Flash_Storage"],
        inplace=True, errors="ignore")

#GPU Brand Feature 
df["Gpu Brand"] = df["Gpu"].apply(lambda x: x.split()[0])
df = df[df["Gpu Brand"] != "ARM"]  # drop rare ARM GPUs

# OS Feature
def cat_os(inp):
    if inp in ["Windows 10", "Windows 7", "Windows 10 S"]:
        return "Windows"
    elif inp in ["macOS", "Mac OS X"]:
        return "Mac"
    else:
        return "Others/No OS/Linux"

df["os"] = df["OpSys"].apply(cat_os)

#columns that can be dropped
df.drop(columns=['Cpu','Cpu Name'],inplace=True) #redundant string features
df.drop(columns=['OpSys'],inplace=True)  #redundant string features
df.drop(columns=['Memory'],inplace=True) #redudant string features
df.drop(columns=['Gpu'],inplace=True)#redundant string features

print(df.head(10))
print(df.select_dtypes(include=[np.number]).corr()["Price"].sort_values(ascending=False)) 