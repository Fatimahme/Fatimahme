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

# First, we ensure that 'Current Subscription Status' is encoded correctly
df_encoded = df.copy()

# Label encode 'Current Subscription Status' (Premium=1, Free=0)
df_encoded['Numeric Subscription Status'] = df_encoded['Current Subscription Status'].apply(lambda x: 1 if x == "Premium" else 0)
df_encoded['Numeric Engagement Trend'] = df_encoded['Engagement Trend'].apply(lambda x: 0 if x == "Decreasing" else (0.5 if x == "Stable" else 1))
# Now, let's remove 'User ID' since it's just an identifier and doesn't contribute to correlation
df_encoded = df_encoded.drop(columns=['User ID','Days to Convert'])

# Select only numeric columns for correlation
df_numeric = df_encoded.select_dtypes(include=['number'])

# Calculate the correlation matrix
correlation_matrix = df_numeric.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix with Encoded Variables")
plt.show()

#when we look at the correlation matrix we understand that there is a High Positive Correlation between Total Watch Time (0.94) and Number of Videos Watched this makes sense, as users who watch more videos tend to have higher watch time. and also a Negative correlation between net promoter score and number of customer support interaction.
#And we can see Subscription Numeris Status that is important for us has Moderate Correlations with Net promete score and a Weak Correlation with Numeric Engagement Trend, Social Shares,number of customer support, total watch and number of watches !
#NOW I want analyze Premium users by Country, Age Group, and Genre
categorical_columns = ['Country', 'Age Group', 'Favorite Genre']

for col in categorical_columns:
    # Get the count of users for each category and status (Premium/Free)
    user_count = pd.crosstab(df[col], df['Current Subscription Status'])
    
    # Calculate the percentage of Premium users out of the total users (Premium + Free) in each segment
    user_count['Premium %'] = (user_count['Premium'] / user_count.sum(axis=1)) * 100
     # Sort the values from least to greatest before plotting to compare easier
    user_count = user_count.sort_values(by="Premium %", ascending=True)
    # Plot the results showing only the Premium percentage
    user_count['Premium %'].plot(kind='bar', color='purple', alpha=0.8, figsize=(8, 5))
    
    plt.title(f"Percentage of Premium Users by {col}")
    plt.ylabel("Percentage of Premium Users")
    plt.xticks(rotation=45)
    plt.show()

#sorrily there is no obvious relation between age, country and genre and having a premium account but perhaps we can use this part of anlysis in A/B test in future!
# Now i want to see more approach of premium account detail
df_premium = df[df['Current Subscription Status'] == 'Premium']

# Define categories to analyze
categories = ['Favorite Genre', 'Country', 'Age Group']

# Function to calculate percentages and plot a pie chart with color scaling
def plot_pie_chart(column):
    # Calculate the percentage of each category in the Premium population
    value_counts = df_premium[column].value_counts(normalize=True) * 100
    
    # Sort values from greatest to least (important for color gradient)
    sorted_values = value_counts.sort_values(ascending=False)

    # Generate a color gradient based on the number of unique values
    colors = sns.color_palette("coolwarm", len(sorted_values))

    # Plot Pie Chart
    plt.figure(figsize=(7, 7))
    plt.pie(sorted_values, labels=sorted_values.index, autopct='%1.1f%%', 
            colors=colors, startangle=140)

    # Title
    plt.title(f"Distribution of {column} in Premium Users", fontsize=14, fontweight='bold')
    plt.show()

# Step 4: Generate plots for each category
for cat in categories:
    plot_pie_chart(cat)
#as we can see all number are really close and differ from 0.1 to 1 but this part of data would be helpful for A/B test

# Calculate days until conversion for premium users to find a relation
df['Days to Convert'] = (df['Date of Premium Subscription'] - df['Registration Date']).dt.days

# Filter only premium users
df_premium = df[df['Current Subscription Status'] == 'Premium']

# Get min and max values dynamically
min_days = df_premium['Days to Convert'].min()
max_days = df_premium['Days to Convert'].max()

# Plot histogram (without KDE)
plt.figure(figsize=(8, 5))
sns.histplot(df_premium['Days to Convert'], bins=30, color="blue", stat="density", alpha=0.6)

# Plot KDE separately with a dynamic range
sns.kdeplot(df_premium['Days to Convert'], color="red", linewidth=2, clip=(min_days, max_days))

plt.xlim(min_days, max_days)  # Dynamically set x-axis limits
plt.title("Distribution of Days to Convert")
plt.xlabel("Days from Registration to Subscription")
plt.show()
#as plot shows we have a high density at the first days of registration and then it becomes stable and then decreasing.

