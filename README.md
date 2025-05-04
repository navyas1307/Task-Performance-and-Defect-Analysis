#Task Performence Quality Analysis Project
As a Data Analyst, the task is to analyze and derive insights from a dataset containing various
features related to tasks, samples, defects, errors, and employee information within a company.
The dataset includes the following features:
XYZ Company operates across multiple departments, each responsible for executing various
tasks critical to the company's operations. These tasks encompass a wide range of activities,
from production processes to administrative functions. However, the company faces challenges
in maintaining consistent task quality and efficiency, as evidenced by varying defect rates and
error occurrences across departments and employee teams.
The dataset provided for analysis contains detailed records of tasks performed, including
timestamps, employee IDs, department IDs, auditor IDs, manager IDs, task types, sample data,
defects, and errors encountered during task execution. Additionally, the dataset includes
supplementary information such as employee names, auditor names, manager names,
department names, and office locations.

## Project Structure

```text
Quality-Analysis-Project/
├── Quality Analysis Dataset.xlsx      # Original Excel data for Power BI
├── data.csv                          # Task, sample, defect, and error metrics
├── manager.csv                       # Employee, manager, department, and location lookup
├── audit.csv                         # Auditor lookup
├── analysis.sql                      # 20 MySQL queries for analysis
├── Quality Analysis.pbix             # Power BI dashboard file
└── README.md                         # Project documentation
```

## Table Schemas

### `data` (from `data.csv`)

| Column         | Type | Description                         |
| -------------- | ---- | ----------------------------------- |
| `Date`         | DATE | Task date (`dd-MM-yyyy`)            |
| `Emp Id`       | INT  | Employee identifier                 |
| `Department Id`| INT  | Department lookup key               |
| `Auditor Id`   | INT  | Auditor lookup key                  |
| `Manager Id`   | INT  | Manager lookup key                  |
| `All Task`     | INT  | Total tasks executed                |
| `Sample`       | INT  | Number of sampled tasks             |
| `Defects`      | INT  | Count of defects identified         |
| `Errors`       | INT  | Count of errors identified          |

### `manager` (from `manager.csv`)

| Column           | Type | Description                    |
| ---------------- | ---- | ------------------------------ |
| `Emp Id`         | INT  | Employee identifier            |
| `Emp Name`       | TEXT | Employee full name             |
| `Manager`        | TEXT | Manager full name              |
| `Manager Id`     | INT  | Unique manager identifier      |
| `Department Id`  | INT  | Department identifier          |
| `Department`     | TEXT | Department name                |
| `Office Location`| TEXT | Geographic office location     |

### `audit` (from `audit.csv`)

| Column         | Type | Description                     |
| -------------- | ---- | ------------------------------- |
| `Auditor Id`   | INT  | Unique auditor identifier       |
| `Auditor Name` | TEXT | Auditor full name               |

## SQL Queries (`analysis.sql`)

All queries use `STR_TO_DATE(\`Date\`, '%d-%m-%Y')` to parse dates. They include:

### 1–5. Monthly Trends
- Task volume
- Sample volume
- Defect rate (% of tasks)
- Error rate (% of tasks)
- Sample coverage (% of tasks sampled)

```sql
-- Example: Monthly Task Volume
SELECT
  YEAR(STR_TO_DATE(`Date`, '%d-%m-%Y')) AS Year,
  MONTH(STR_TO_DATE(`Date`, '%d-%m-%Y')) AS Month,
  SUM(`All Task`) AS Total_Tasks
FROM data
GROUP BY Year, Month
ORDER BY Year, Month;
```

### Departmental Analysis
- Total tasks by department
- Defect & error rates by department
- Quality % (no defects or errors)

### Employee Performance
- Top employees by task volume, defects, errors
- Defect-to-task & error-to-task ratios

### Manager & Auditor Metrics
- Manager defect rate & quality %
- Auditor defect detection & error detection rates

### Location Analysis
- Defect & error rates by office location

### Sampling Insights
- Defect rate within sampled tasks over time

### Dept × Month Dashboard
- Cross-tab of tasks, defects, and errors by department and month

## Power BI Dashboard (`Quality Analysis.pbix`)

Built directly on **`Quality Analysis Dataset.xlsx`**:

1. **Get Data > Excel**: Import `Quality Analysis Dataset.xlsx`.
2. **Power Query**:
   - Parse `Date` as `dd-MM-yyyy`.
   - Rename fields for clarity.
3. **Model**:
   - If using split tables, define relationships:
     - `data[Emp Id]` → `manager[Emp Id]`
     - `data[Manager Id]` → `manager[Manager Id]`
     - `data[Auditor Id]` → `audit[Auditor Id]`
4. **DAX Measures**:
   ```DAX
   All task Total
All task Total = SUM('Data Sampling'[All Task])
Defects Percentages
Defect % = DIVIDE(SUM('Data Sampling'[Defects]), SUM('Data Sampling'[Sample]),
BLANK())
Do Not Copy
Defects Total
Defects Total = SUM('Data Sampling'[Defects])
Errors Percentage
Error % = DIVIDE(SUM('Data Sampling'[Errors]), SUM('Data Sampling'[Sample]),
BLANK())
Errors Total
Errors Total = SUM('Data Sampling'[Errors])
Quality Score
Quality Score = IF([Defect %] = BLANK(), BLANK(), 1 - [Defect %])
Sample Percentages
Sample % = DIVIDE(SUM('Data Sampling'[Sample]), SUM('Data Sampling'[All Task]),
BLANK())
Sample Total
Sample Total = SUM('Data Sampling'[Sample])
   ```
5. **Visualizations**:
   - Line charts for trends
   - Bar charts for top performers
   - Matrices for detailed breakdowns

## Getting Started

### MySQL Setup
```bash
mysql -u <username> -p
CREATE DATABASE quality_analysis;
USE quality_analysis;
LOAD DATA INFILE 'data.csv' INTO TABLE data
  FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 ROWS;
# Repeat for manager.csv and audit.csv
SOURCE analysis.sql;
```

### Power BI
1. Open `Quality Analysis.pbix` in Power BI Desktop.
2. Refresh data source (Excel).  



