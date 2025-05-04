use quality_analysis;
#1.Monthly Task Volume
SELECT
  YEAR(STR_TO_DATE(d.`Date`, '%d-%m-%Y')) AS Year,
  MONTH(STR_TO_DATE(d.`Date`, '%d-%m-%Y')) AS Month,
  SUM(d.`All Task`) AS Total_Tasks
FROM data AS d
GROUP BY Year, Month
ORDER BY Year, Month;

#2.Monthly Sample Volume
SELECT
  YEAR(STR_TO_DATE(d.`Date`, '%d-%m-%Y')) AS Year,
  MONTH(STR_TO_DATE(d.`Date`, '%d-%m-%Y')) AS Month,
  SUM(d.`Sample`) AS Total_Samples
FROM data AS d
GROUP BY Year, Month
ORDER BY Year, Month;

#3.Monthly Defect Rate (% of tasks)
SELECT
  YEAR(STR_TO_DATE(d.`Date`, '%d-%m-%Y')) AS Year,
  MONTH(STR_TO_DATE(d.`Date`, '%d-%m-%Y')) AS Month,
  SUM(d.`Defects`)/SUM(d.`All Task`) * 100 AS Defect_Rate_Percent
FROM data AS d
GROUP BY Year, Month
ORDER BY Year, Month;

#4.Monthly Error Rate (% of tasks)
SELECT
  YEAR(STR_TO_DATE(d.`Date`, '%d-%m-%Y')) AS Year,
  MONTH(STR_TO_DATE(d.`Date`, '%d-%m-%Y')) AS Month,
  SUM(d.`Errors`)/SUM(d.`All Task`) * 100 AS Error_Rate_Percent
FROM data AS d
GROUP BY Year, Month
ORDER BY Year, Month;

#5.Monthly Sample Coverage (% of tasks sampled)
SELECT
  YEAR(STR_TO_DATE(d.`Date`, '%d-%m-%Y')) AS Year,
  MONTH(STR_TO_DATE(d.`Date`, '%d-%m-%Y')) AS Month,
  SUM(d.`Sample`)/SUM(d.`All Task`) * 100 AS Sample_Coverage_Percent
FROM data AS d
GROUP BY Year, Month
ORDER BY Year, Month;

#6.Department-wise Task Volume
SELECT
  m.`Department`    AS Department,
  SUM(d.`All Task`) AS Total_Tasks
FROM data AS d
JOIN manager AS m
  ON d.`Emp Id` = m.`Emp Id`
GROUP BY m.`Department`
ORDER BY Total_Tasks DESC;

#7.Department-wise Defect & Error Rates
SELECT
  m.`Department`    AS Department,
  SUM(d.`Defects`)/SUM(d.`All Task`) * 100 AS Defect_Rate,
  SUM(d.`Errors`)/SUM(d.`All Task`)  * 100 AS Error_Rate
FROM data AS d
JOIN manager AS m
  ON d.`Emp Id` = m.`Emp Id`
GROUP BY m.`Department`
ORDER BY Defect_Rate DESC;

#8.Top 5 Employees by Task Volume
SELECT
  m.`Emp Name`       AS Employee,
  SUM(d.`All Task`)  AS Total_Tasks
FROM data AS d
JOIN manager AS m
  ON d.`Emp Id` = m.`Emp Id`
GROUP BY m.`Emp Name`
ORDER BY Total_Tasks DESC
LIMIT 5;

#9.Top 5 Employees by Defect Count
SELECT
  m.`Emp Name`      AS Employee,
  SUM(d.`Defects`)  AS Total_Defects
FROM data AS d
JOIN manager AS m
  ON d.`Emp Id` = m.`Emp Id`
GROUP BY m.`Emp Name`
ORDER BY Total_Defects DESC
LIMIT 5;

#10.Top 5 Employees by Error Count
SELECT
  m.`Emp Name`     AS Employee,
  SUM(d.`Errors`)  AS Total_Errors
FROM data AS d
JOIN manager AS m
  ON d.`Emp Id` = m.`Emp Id`
GROUP BY m.`Emp Name`
ORDER BY Total_Errors DESC
LIMIT 5;

#11.Employee Defect-to-Task Ratio
SELECT
  m.`Emp Name`                                   AS Employee,
  SUM(d.`Defects`)/SUM(d.`All Task`)              AS Defect_to_Task_Ratio
FROM data AS d
JOIN manager AS m
  ON d.`Emp Id` = m.`Emp Id`
GROUP BY m.`Emp Name`
ORDER BY Defect_to_Task_Ratio DESC;

#12.Manager-wise Defect Rate (%)
SELECT
  m.`Manager`                                   AS Manager,
  SUM(d.`Defects`)/SUM(d.`All Task`) * 100       AS Defect_Rate_Percent
FROM data AS d
JOIN manager AS m
  ON d.`Manager Id` = m.`Manager Id`
GROUP BY m.`Manager`
ORDER BY Defect_Rate_Percent DESC;

#13.Auditor-wise Defect Detection Rate (%)

SELECT
  a.`Auditor Name`                               AS Auditor,
  SUM(d.`Defects`)/SUM(d.`All Task`) * 100        AS Defect_Detection_Percent
FROM data AS d
JOIN audit AS a
  ON d.`Auditor Id` = a.`Auditor Id`
GROUP BY a.`Auditor Name`
ORDER BY Defect_Detection_Percent DESC;

#14.Auditor-wise Error Detection Rate (%)
SELECT
  a.`Auditor Name`                              AS Auditor,
  SUM(d.`Errors`)/SUM(d.`All Task`) * 100        AS Error_Detection_Percent
FROM data AS d
JOIN audit AS a
  ON d.`Auditor Id` = a.`Auditor Id`
GROUP BY a.`Auditor Name`
ORDER BY Error_Detection_Percent DESC;

#15.Location-wise Defect & Error Rates
SELECT
  m.`Office Location`                           AS Location,
  SUM(d.`Defects`)/SUM(d.`All Task`) * 100       AS Defect_Rate,
  SUM(d.`Errors`)/SUM(d.`All Task`)  * 100       AS Error_Rate
FROM data AS d
JOIN manager AS m
  ON d.`Emp Id` = m.`Emp Id`
GROUP BY m.`Office Location`
ORDER BY Defect_Rate DESC;

















