import pandas as pd

# 1) Load the Dataset
df = pd.read_csv('https://raw.githubusercontent.com/ArchanaInsights/Datasets/main/marketing_campaign.csv')
print(df)

# 2) Descriptive Analysis

# Basic Structure:

# ➢ Print the first few rows of the dataset to get an overview of the data.
print(df.head())

# ➢ Obtain the number of rows and columns in the dataset.
print(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")

# ➢ Get a concise summary of the dataset, including the data types and non-null values.
print(df.info())

# ➢ Generate descriptive statistics for numerical columns.
print(df.describe())


# Data Exploration:

# ➢ Print the number of unique Campaign_ID values in the dataset.
print(f"Number of unique Campaign_ID values: {df['Campaign_ID'].nunique()}")

# ➢ List the unique values of the Location and Customer_Segment columns.
print(f"Unique Location values: {df['Location'].unique()}")
print(f"Unique Customer_Segment values: {df['Customer_Segment'].unique()}")

# ➢ Count the occurrences of each category in the Campaign_Type and Channel_Used and columns.
print(f"Count of occurences in Campaign_Type :\n{df['Campaign_Type'].value_counts()}")
print(f"Count of occurences in Channel_Used :\n{df['Channel_Used'].value_counts()}")

# 3) Exploratory Data Analysis (EDA) and Visualization

# ➢ Plot a scatter plot to visualize the relationship between Acquisition_Cost and ROI.
import matplotlib.pyplot as plt

plt.scatter(df['Acquisition_Cost'], df['ROI'])
plt.xlabel('Acquisition Cost')
plt.ylabel('ROI')
plt.title('Relationship between Acquisition Cost and ROI')
plt.show()

# ➢ Create a bar chart to visualize the average Conversion_Rate for different Channel_Used, categorized by Campaign_Type.
plt.figure(figsize=(10, 6))
df_grouped = df.groupby(['Channel_Used', 'Campaign_Type'])['Conversion_Rate'].mean().unstack()
df_grouped.plot(kind='bar', stacked=True)
plt.xlabel('Channel Used')
plt.ylabel('Average Conversion Rate')   
plt.title('Average Conversion Rate for different Channel Used, categorized by Campaign Type')
plt.legend(title='Campaign Type')
plt.show()


# ➢ Visualize the distribution of Engagement_Score across different Campaign_Type using a box plot.

plt.figure(figsize=(10, 6))
import seaborn as sns   
plot = sns.boxplot(x='Campaign_Type', y='Engagement_Score', data=df)
plot.set_title('Distribution of Engagement Score across different Campaign Type')
plt.show()

# ➢ Analyze the average ROI by Company using a bar chart to compare the profitability of campaigns conducted by different companies.

plt.figure(figsize=(10, 6))
plot = sns.barplot(x='Company', y='ROI', data=df, ci=None)
plot.set_title('Average ROI by Company')
plt.xlabel('Company')
plt.ylabel('Average ROI')
plt.show()

# ➢ Examine the correlation between Engagement_Score and Conversion_Rate using a heatmap.

plt.figure(figsize=(8, 6))
correlation_matrix = df[['Engagement_Score', 'Conversion_Rate']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation between Engagement Score and Conversion Rate')
plt.show()

# Customer Segmentation:

# ➢ Create a count plot to visualize the distribution of Target_Audience.

plt.figure(figsize=(10, 6))
plot = sns.countplot(x='Target_Audience', data=df)
plot.set_title('Distribution of Target Audience')
plt.xlabel('Target Audience')
plt.ylabel('Count')
plt.show()

# ➢ Identify which Customer_Segment has the highest Conversion_Rate for each Language using a bar chart.

plt.figure(figsize=(12, 6))
df_grouped = df.groupby(['Language', 'Customer_Segment'])['Conversion_Rate'].mean().unstack()
df_grouped.plot(kind='bar', stacked=True)
plt.title('Highest Conversion Rate for each Language by Customer Segment')
plt.xlabel('Language')
plt.ylabel('Average Conversion Rate')
plt.legend(title='Customer Segment')
plt.show()

# ➢ Visualize the distribution of Acquisition_Cost across each Customer_Segment,categorized by Channel_Used, using a box plot.

plt.figure(figsize=(12, 6))
plot = sns.boxplot(x='Customer_Segment', y='Acquisition_Cost', hue='Channel_Used', data=df)
plot.set_title('Distribution of Acquisition Cost across each Customer Segment, categorized by Channel Used')
plt.xlabel('Customer Segment')
plt.ylabel('Acquisition Cost')
plt.legend(title='Channel Used')
plt.show()

# ➢ Analyze average Conversion_Rate by Language using a bar chart to compare the effectiveness of campaigns conducted in different languages.

plt.figure(figsize=(10, 6))
plot = sns.barplot(x='Language', y='Conversion_Rate', data=df, ci=None)
plot.set_title('Average Conversion Rate by Language')
plt.xlabel('Language')
plt.ylabel('Average Conversion Rate')
plt.show()

# Channel Effectiveness

# ➢ Compare the Engagement_Score for different Channels_Used, segmented by Campaign_Type, using a bar chart.

plt.figure(figsize=(12, 6))
df_grouped = df.groupby(['Channel_Used', 'Campaign_Type'])['Engagement_Score'].mean().unstack()
df_grouped.plot(kind='bar', stacked=True)
plt.title('Engagement Score for different Channels Used, segmented by Campaign Type')
plt.xlabel('Channel Used')
plt.ylabel('Average Engagement Score')
plt.legend(title='Campaign Type')
plt.show()

# ➢ Show the distribution of total ROI across different Channels_Used using a pie chart.

plt.figure(figsize=(8, 8))
roi_by_channel = df.groupby('Channel_Used')['ROI'].sum()
plt.pie(roi_by_channel, labels=roi_by_channel.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Total ROI across different Channels Used')
plt.show()

# ➢ Plot a scatter plot to show the relationship between Clicks and Impressions for each Channel_Used.

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Impressions', y='Clicks', hue='Channel_Used', data=df)
plt.title('Relationship between Clicks and Impressions for each Channel Used')
plt.xlabel('Impressions')
plt.ylabel('Clicks')
plt.legend(title='Channel Used')
plt.show()

# Time-Based Analysis:

# ➢ Plot the distribution of Duration using a histogram.
plt.figure(figsize=(10, 6))
plt.hist(df['Duration'], bins=30, edgecolor='black')
plt.title('Distribution of Duration')
plt.xlabel('Duration')
plt.ylabel('Count')
plt.show()

# ➢ Analyze how the overall Conversion_Rate has changed over Date for each Company using a line chart.

plt.figure(figsize=(12, 6))
df_grouped = df.groupby(['Date', 'Company'])['Conversion_Rate'].mean().unstack()
df_grouped.plot()
plt.title('Overall Conversion Rate over Date for each Company')
plt.xlabel('Date')
plt.ylabel('Average Conversion Rate')
plt.legend(title='Company')
plt.show()


# ➢ Examine the trend of Engagement_Score over Date with a line chart.
plt.figure(figsize=(12, 6))
plot = sns.lineplot(x='Date', y='Engagement_Score', data=df)
plot.set_title('Trend of Engagement Score over Date')
plt.xlabel('Date')
plt.ylabel('Engagement Score')
plt.show()

# Geographic Analysis

# ➢ Determine which location has the highest Acquisition_Cost using a bar chart.

plt.figure(figsize=(10, 6))
df_grouped = df.groupby('Location')['Acquisition_Cost'].mean().sort_values(ascending=False)
df_grouped.plot(kind='bar')
plt.title('Average Acquisition Cost by Location')
plt.xlabel('Location')
plt.ylabel('Average Acquisition Cost')
plt.show()

# ➢ Visualize the Conversion_Rate by different Location, categorized by Target_Audience, using a bar chart.

plt.figure(figsize=(12, 6))
df_grouped = df.groupby(['Location', 'Target_Audience'])['Conversion_Rate'].mean().unstack()
df_grouped.plot(kind='bar', stacked=True)
plt.title('Conversion Rate by Location, categorized by Target Audience')
plt.xlabel('Location')
plt.ylabel('Average Conversion Rate')
plt.legend(title='Target Audience')
plt.show()

# ➢ Illustrate the proportion of ROI by Location using a pie chart.

plt.figure(figsize=(8, 8))
roi_by_location = df.groupby('Location')['ROI'].sum()
plt.pie(roi_by_location, labels=roi_by_location.index, autopct='%1.1f%%', startangle=140)
plt.title('Proportion of ROI by Location')
plt.show()




