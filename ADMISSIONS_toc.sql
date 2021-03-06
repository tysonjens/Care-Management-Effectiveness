/****** Script for SelectTopNRows command from SSMS  ******/
SELECT fca.[Patient_SK]
      --,fca.[Encounter_Type_SK]
      --,fca.[Fund_Type_SK]
      ,fca.[Claim_Number]
      --,fca.[Line_Item]
      --,fca.[Service_Period_SK]
      --,fca.[Date_Service_From_SK]
      --,fca.[Date_Service_To_SK]
      --,fca.[Process_Date_SK]
      --,fca.[Healthplan_SK]
      --,fca.[Product_SK]
      --,fca.[Att_Site_Market_Hierarchy_SK]
      --,fca.[Att_PCP_SK]
      ,fca.[Place_of_Service_SK]
      --,fca.[Vendor_SK]
      ,fca.[Rendering_Provider_SK]
      ,fca.[Specialty_SK]
      ,fca.[ICD_DX_Group_SK]
      --,fca.[ICD_PX_Group_SK]
      ,fca.[Procedure_Code_SK]
      --,fca.[MS_DRG_SK]
      ,fca.[Modifier_Group_SK]
      ,fca.[Revenue_Code_SK]
      --,fca.[Billed_Type_Key]
      ,fca.[Elig_Flag]
      ,fca.[Age_Group_SK]
      --,fca.[AHRQ_Adm_Flag]
      --,fca.[PQI_Measure_SK]
      --,fca.[DOS_Meq_Amount]
      --,fca.[CY_Meq_Amount]
      --,fca.[APP_Days]
      --,fca.[APP_Amount]
      --,fca.[APP_Unit]
      --,fca.[Billed_Unit]
      --,fca.[Billed_Amount]
      --,fca.[RVU]
      --,fca.[Paid_Amount]
      --,fca.[Create_Date]
      --,fca.[Source_System_SK]
      --,fca.[RiskTypeSK]
      --,fca.[RiskSharingPercent]
      --,fca.[ClaimTypeSK]
      ,fca.[Encounter_Count]
      ,fca.[ReAdmits]
      ,fca.[DurationOfStay]
      ,fca.[ERWithAdmits]
      ,fca.[AvoidableER]
      ,fca.[FollowUps]
	  ,drc.Revenue_Code_Description
	  ,dd.[Date_BK] as admit_date
	  ,dd2.[Date_BK] as discharge_date
	  ,dpos.[Place_of_Service_Name]
	  ,dp.EMPI
  FROM [NATIONAL_ANALYTICS].[dbo].[FACT_CLAIM_ACO] fca
  JOIN [NATIONAL_ANALYTICS].[dbo].[DIM_PATIENT] dp ON dp.[Patient_SK] = fca.Patient_SK
  JOIN [NATIONAL_ANALYTICS].[dbo].[DIM_REVENUE_CODE] drc ON drc.Revenue_Code_SK = fca.Revenue_Code_SK
  JOIN [NATIONAL_ANALYTICS].[dbo].[DIM_DATE] dd ON dd.Date_SK = fca.Date_Service_From_SK
  JOIN [NATIONAL_ANALYTICS].[dbo].[DIM_DATE] dd2 ON dd2.Date_SK = fca.Date_Service_To_SK
  JOIN [NATIONAL_ANALYTICS].[dbo].[DIM_PLACE_OF_SERVICE] dpos ON dpos.Place_of_Service_SK = fca.Place_of_Service_SK
  WHERE [Encounter_Count] = 'Admit'
  AND [Line_Item] = 1
  AND [Date_Service_From_SK] > 20160801