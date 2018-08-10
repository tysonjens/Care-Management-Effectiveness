/****** Script for SelectTopNRows command from SSMS  ******/
WITH Distinctmembers AS
(SELECT dp.[Patient_SK]
      ,dp.[EMPI]
      ,dp.[Patient_Name]
      ,dp.[DOB]
      ,dp.[Sex]
      ,dp.[Race]
      ,dp.[MRN]
	  ,dp.[States]
      ,dp.[Current_Member]
	  ,dp.[End_Date] 
	  ,fe.[Elig_End_Date] 
	  ,dh.[HealthPlan_Name],
	  ROW_NUMBER() OVER(PARTITION BY dp.[Patient_SK] ORDER BY fe.[Elig_End_Date] DESC) AS 'rownum'
  FROM [NATIONAL_ANALYTICS].[dbo].[DIM_PATIENT] dp
  JOIN [NATIONAL_ANALYTICS].[dbo].[FACT_ELIGIBILITY] fe ON dp.[Patient_SK] = fe.[Patient_SK]
  JOIN [NATIONAL_ANALYTICS].[dbo].[DIM_HEALTHPLAN] dh ON fe.[HealthPlan_SK] = dh.[HealthPlan_SK]
  WHERE dp.[Current_Member] = 'Y'
  AND dp.[States] = 'CA'
  AND fe.[Elig_End_Date] > '2018-01-01'
)
SELECT *
FROM Distinctmembers
WHERE RowNum = 1