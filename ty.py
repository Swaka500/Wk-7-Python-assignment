# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Dataset loaded from sklearn
try:
    iris = load_iris(as_frame=True)
    df = iris.frame
    print("Dataset loaded successfully.")
except Exception as e:
    print("Error loading dataset:", e)

# The first few rows displayed
print("\nFirst 5 rows:")
print(df.head())

# Explore structure of the dataset
print("\nDataset Information:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Basic Data Analysis
print("\nBasic Statistics:")
print(df.describe())

# Group by categorical column ('target') and compute mean
group_means = df.groupby('target').mean()
print("\nMean values grouped by target (species):")
print(group_means)

# Identify patterns
print("\nObservations from Grouping:")
print("- Species 0 tends to have smaller petal measurements.")
print("- Species 2 has the highest petal width and length on average.")

# Data Visualization

# 1. Line Chart - Mean Sepal Length per Species
plt.figure(figsize=(8,5))
group_means['sepal length (cm)'].plot(marker='o')
plt.title('Mean Sepal Length per Species')
plt.xlabel('Species (target)')
plt.ylabel('Mean Sepal Length (cm)')
plt.grid(True)
plt.show()

# 2. Bar Chart - Average Petal Length per Species
plt.figure(figsize=(8,5))
group_means['petal length (cm)'].plot(kind='bar', color='lightblue')
plt.title('Average Petal Length per Species')
plt.xlabel('Species (target)')
plt.ylabel('Average Petal Length (cm)')
plt.xticks(rotation=0)
plt.show()

# 3. Histogram - Distribution of Sepal Width
plt.figure(figsize=(8,5))
plt.hist(df['sepal width (cm)'], bins=15, color='green', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter Plot - Sepal Length vs Petal Length
plt.figure(figsize=(8,5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target', data=df)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# Findings

print("\nFinal Observations:")
print("- Petal length and sepal length are positively correlated.")
print("- Sepal width is fairly normally distributed.")
print("- Different species have visibly different ranges for petal length.")

    # THE END
