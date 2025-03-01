import pandas as pd
import numpy as np
from google.colab import files
import seaborn as sns
import matplotlib.pyplot as plt

# Upload CSV file
uploaded = files.upload()

# Get the filename from device to google colab
filename = list(uploaded.keys())[0]

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(filename)

# Display basic info
print("Data Overview:")
print(df.info())

# Convert date columns to datetime format
date_columns = ['Registration Date', 'Date of Premium Subscription']
for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Check for missing values and i find out missing data is for free accounts because they have no date of premiumm subs so we can not drop them because whole data of free account will be deleted
print("\nMissing Values:\n", df.isnull().sum())

df = df.copy()  # Prevents chained indexing issues

# Convert categorical variables to category dtype
categorical_cols = ['Current Subscription Status', 'Primary Device', 'Age Group', 'Country', 'Favorite Genre', 'Engagement Trend']
for col in categorical_cols:
    df.loc[:, col] = df[col].astype("category")  # Explicitly using .loc to avoid warnings

# Display first few rows
df.head(10)
# Define color mapping
custom_palette = {"Premium": "red", "Free": "blue"}

# Define category order explicitly
category_order = ["Free", "Premium"]

# Plot with explicit category order
plt.figure(figsize=(8, 5))
sns.countplot(
    x="Current Subscription Status",
    data=df,
    hue="Current Subscription Status",
    palette=custom_palette,
    order=category_order,  # Force both "Free" and "Premium" to appear
    legend=False
)
plt.title("Subscription Status Distribution")
plt.xlabel("Subscription Type")
plt.ylabel("Count")
plt.show()
# Define the metrics for boxplot analysis
metrics = ['Number of Videos Watched', 'Total Watch Time', 'Number of Customer Support Interactions', 'Net Promoter Score', 'Social Shares']

plt.figure(figsize=(12, 6))

# Loop through the metrics and create a subplot for each
for i, col in enumerate(metrics, 1):
    plt.subplot(2, 3, i)
    
    # Fix the boxplot by assigning 'hue' and using the palette
    sns.boxplot(x="Current Subscription Status", y=col, data=df, hue="Current Subscription Status", palette="coolwarm", legend=False)
    
    plt.title(f"{col} by Subscription Status")
# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.9)
plt.tight_layout()
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your original DataFrame

# First, we ensure that 'Current Subscription Status' is encoded correctly
df_encoded = df.copy()

# Label encode 'Current Subscription Status' (Premium=1, Free=0)
df_encoded['Subscription_Status_Numeric'] = df_encoded['Current Subscription Status'].apply(lambda x: 1 if x == "Premium" else 0)

# Now, let's remove 'User ID' since it's just an identifier and doesn't contribute to correlation
df_encoded = df_encoded.drop(columns=['User ID'])

# Select only numeric columns for correlation
df_numeric = df_encoded.select_dtypes(include=['number'])

# Calculate the correlation matrix
correlation_matrix = df_numeric.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix with Encoded Variables")
plt.show()
#when we look at the correlation matrix we understand that there is a High Positive Correlation between Total Watch Time (0.94) and Number of Videos Watched this makes sense, as users who watch more videos tend to have higher watch time.
#And we can see Subscription Numeris Status that is important for us has Moderate Correlations with Net promete score and a Weak Correlation with Social Shares,number of customer support, total watch and number of watches !
