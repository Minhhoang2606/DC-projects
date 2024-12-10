'''
Building Financial Reports
Author: Henry Ha
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the balance sheet
balance_sheet = pd.read_excel(R"..\Building Financial Reports\data\Balance_Sheet.xlsx")

# Load the income statement
income_statement = pd.read_excel(R"..\Building Financial Reports\data\Income_Statement.xlsx")

# Display the first few rows of each dataset
print(balance_sheet.head())
print(income_statement.head())

# Get a summary of the data
print(balance_sheet.info())
print(income_statement.info())

# Check the unique companies in the dataset
unique_companies = balance_sheet["company"].unique()
print(unique_companies)

#TODO : Data cleaning and preprocessing

# Handlle missing values
balance_sheet['Inventory'].fillna(balance_sheet['Inventory'].mean(), inplace=True)
balance_sheet['Short Term Investments'].fillna(balance_sheet['Short Term Investments'].mean(), inplace=True)

# Remove unnecessary columns
balance_sheet.drop(columns=['Unnamed: 0'], inplace=True)
income_statement.drop(columns=['Unnamed: 0'], inplace=True)

# Verify data types
balance_sheet.info()
income_statement.info()

# Consistency Check
# Unique years in each dataset
print(balance_sheet['Year'].unique())
print(income_statement['Year'].unique())

# Unique companies in each dataset
print(balance_sheet['company'].unique())
print(income_statement['company'].unique())

#TODO: Analyse financial data

# Balance sheet analysis
balance_sheet['Debt-to-Equity'] = balance_sheet['Total Liab'] / balance_sheet['Total Stockholder Equity']
print(balance_sheet[['company', 'Year', 'Debt-to-Equity']])


# Filter the data to ensure each company has exactly 4 years
companies = balance_sheet['company'].unique()
filtered_data = balance_sheet.groupby('company').head(4)  # Keep 4 rows per company

# Create a color palette for the bars (one color per year per company)
years = filtered_data['Year'].unique()
colors = plt.cm.tab20(range(len(years)))  # Generate a distinct color for each year

# Plot the data
plt.figure(figsize=(14, 8))
for i, company in enumerate(companies):
    company_data = filtered_data[filtered_data['company'] == company]
    bar_positions = [i + (j / 5) for j in range(4)]  # Offset bars slightly for visibility
    plt.bar(
        bar_positions,
        company_data['Debt-to-Equity'],
        color=[colors[j] for j in range(len(company_data))],
        width=0.15,
        label=f"{company} {list(company_data['Year'])}",
    )

# Add labels and title
plt.xlabel('Company')
plt.ylabel('Debt-to-Equity')
plt.title('Debt-to-Equity Ratio by Company (Grouped by Year)')
plt.xticks(range(len(companies)), companies, rotation=90)
plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Profitability Trends, Income Statement Analysis:

# Calculate the Gross Margin and Operating Margin to evaluate profitability
income_statement['Gross Margin'] = income_statement['Gross Profit'] / income_statement['Total Revenue']
income_statement['Operating Margin'] = income_statement['Operating Income'] / income_statement['Total Revenue']
print(income_statement[['company', 'Year', 'Gross Margin', 'Operating Margin']])

# Gross Margin Plot
plt.figure(figsize=(12, 8))
for company in income_statement['company'].unique():
    company_data = income_statement[income_statement['company'] == company]
    plt.plot(
        company_data['Year'],
        company_data['Gross Margin'],
        marker='o',
        label=f'{company}'
    )
plt.xlabel('Year')
plt.ylabel('Gross Margin')
plt.title('Gross Margin Trends (All Companies)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Company')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Gross Margin Plot with Labels
plt.figure(figsize=(12, 8))
for company in income_statement['company'].unique():
    company_data = income_statement[income_statement['company'] == company]
    plt.plot(
        company_data['Year'],
        company_data['Gross Margin'],
        marker='o',
        label=f'{company}'
    )
    # Add company name at the endpoint
    plt.text(
        company_data['Year'].iloc[-1],  # X coordinate: last year
        company_data['Gross Margin'].iloc[-1],  # Y coordinate: last margin
        company,
        fontsize=8,
        ha='left',
        va='center'
    )

plt.xlabel('Year')
plt.ylabel('Gross Margin')
plt.title('Gross Margin Trends (All Companies)')
plt.tight_layout()
plt.show()

# Group by company type and calculate the mean gross margin
gross_margin_by_type = income_statement.groupby('comp_type')['Gross Margin'].mean().reset_index()

# Plot the data
plt.figure(figsize=(10, 6))
bars = plt.bar(gross_margin_by_type['comp_type'], gross_margin_by_type['Gross Margin'], color='skyblue')
plt.xlabel('Company Type')
plt.ylabel('Average Gross Margin')
plt.title('Comparison of Gross Margin by Company Type')
plt.xticks(rotation=45)

# Add value labels on each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Operating Margin Plot with Labels
plt.figure(figsize=(12, 8))

for company in income_statement['company'].unique():
    company_data = income_statement[income_statement['company'] == company]
    # Plot the line for each company
    plt.plot(
        company_data['Year'],
        company_data['Operating Margin'],
        marker='x',
        label=f'{company}'
    )
    # Add company name at the endpoint
    plt.text(
        company_data['Year'].iloc[-1],  # X coordinate: last year
        company_data['Operating Margin'].iloc[-1],  # Y coordinate: last margin
        company,
        fontsize=8,
        ha='left',
        va='center'
    )

plt.xlabel('Year')
plt.ylabel('Operating Margin')
plt.title('Operating Margin Trends (All Companies)')
plt.tight_layout()
plt.show()

# Group the data by 'Year' and 'comp_type' to calculate the mean Operating Margin for each type
operating_margin_by_type = income_statement.groupby(['Year', 'comp_type'])['Operating Margin'].mean().reset_index()

# Plotting the Operating Margin Trends by Company Type
plt.figure(figsize=(12, 8))
for comp_type in operating_margin_by_type['comp_type'].unique():
    comp_type_data = operating_margin_by_type[operating_margin_by_type['comp_type'] == comp_type]
    plt.plot(
        comp_type_data['Year'],
        comp_type_data['Operating Margin'],
        marker='x',
        label=f'{comp_type}'
    )

plt.xlabel('Year')
plt.ylabel('Average Operating Margin')
plt.title('Operating Margin Trends by Company Type')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Company Type')
plt.tight_layout()
plt.show()

# Group by year to calculate total revenue for all companies
yearly_revenue = income_statement.groupby('Year')['Total Revenue'].sum()

# Plot the total revenue by year
plt.figure(figsize=(10, 6))
plt.plot(yearly_revenue.index, yearly_revenue.values, marker='o', linestyle='-', label='Total Revenue')
plt.xlabel('Year')
plt.ylabel('Total Revenue (in billions)')
plt.title('Year-on-Year Total Revenue Trends')
plt.grid(True)
plt.tight_layout()
plt.show()

# Cross-Company Analysis

# Filter the Debt-to-Equity data for visualization
plt.figure(figsize=(12, 6))
sns.barplot(
    x='company',
    y='Debt-to-Equity',
    data=balance_sheet,
    ci=None,
    palette='viridis'
)
plt.xlabel('Company')
plt.ylabel('Debt-to-Equity Ratio')
plt.title('Debt-to-Equity Ratios by Company')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Gross Margin by Company
plt.figure(figsize=(12, 6))
sns.barplot(
    x='company',
    y='Gross Margin',
    data=income_statement,
    ci=None,
    palette='coolwarm'
)
plt.xlabel('Company')
plt.ylabel('Gross Margin')
plt.title('Gross Margin by Company')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Assuming 'income_statement' has the calculated 'Gross Margin' and 'Debt-to-Equity Ratio'
# Rename for plotting
data = income_statement.copy()
data['profitability_ratio'] = data['Gross Margin']
data['leverage_ratio'] = data['Debt-to-Equity']

# Create subplots for each company type
unique_types = data['comp_type'].unique()
fig, axes = plt.subplots(1, len(unique_types), figsize=(15, 5), sharey=True)

for ax, comp_type in zip(axes, unique_types):
    subset = data[data['comp_type'] == comp_type]
    sns.regplot(x='leverage_ratio', y='profitability_ratio', data=subset, ax=ax, ci=95, scatter_kws={'alpha': 0.7})
    ax.set_title(f'{comp_type.capitalize()} Companies')
    ax.set_xlabel('Leverage Ratio (Debt-to-Equity)')
    ax.set_ylabel('Profitability Ratio (Gross Margin)')

plt.tight_layout()
plt.show()
