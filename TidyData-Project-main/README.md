# TidyData-Project

This is my project for my second portfolio update in MDSC 20009. I will be practicing data cleaning, tidying, and visualization skills.

Project Overview:
- This project analyzes the 2008 Olympic medalists dataset using tidy data principles to clean, transform, and visualize the data. The goal is to reshape the dataset into a more structured format where:
  - Each variable has its own column.
  - Each observation has its own row.
  - Each type of observational unit is in a separate table.

Setup and Dependencies: 
To run this notebook, you will need the following dependencies:
- Python
- Pandas
- Matplotlib
- Seaborn

Dataset Description
Source: The dataset contains information about the 2008 Olympic medalists.

Pre-processing Steps:
1. Converted wide-format data to long format using pd.melt().
2. Extracted gender and sport from the combined column.
3. Removed unnecessary columns and missing values.
4. Ensured proper data types for consistency.

**Key Features**
Data Cleaning: Transforming and reshaping the dataset using tidy data principles.

Visualizations:
1. Top 10 sports with most medals: A bar chart showcasing the sports that awarded the highest number of medals.
2. Medal distribution by gender: A count plot analyzing how medals were distributed between male and female athletes.
3. Pivot Table Analysis: Aggregating medal counts by sport and medal type to identify the most successful sports.

References:
- Pandas Cheat Sheet
- Tidy Data Principles by Hadley Wickham

**Visual Examples**
Top 10 Sports by Medal Count
<img width="1021" alt="Screenshot 2025-03-17 at 6 02 32 PM" src="https://github.com/user-attachments/assets/7004886d-d970-426b-a645-8f7901f34f37" />

Medal Distribution by Gender
<img width="757" alt="Screenshot 2025-03-17 at 6 02 42 PM" src="https://github.com/user-attachments/assets/16e151ad-49dc-4e7f-981d-657bd186db80" />


