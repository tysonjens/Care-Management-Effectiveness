/****** Script for SelectTopNRows command from SSMS  ******/
SELECT
	  P.[EMPI]
	  ,P.[PTNT_DK]
	  ,S.[PTNT_DK]
	  ,S.[RGON_NM]
	  ,S.[HP_NM]
	  ,S.[LOB_SUB_CGY]
	  ,DP.[DOB]
	  ,DP.[Sex]
	  ,P.[PRGM_NM]
	  ,P.[CRT_TMS]
	  ,P.[END_TMS]
	  ,P.[TNT_MKT_BK]
      ,P.[PRGM_STS]
	  ,P.[PRGM_STOP_RSN]
	  
  FROM [CIM_RPT].[dbo].[PTNT_PRGM] as P
  
  Join [CIM_RPT].[dbo].[PTNT_RPT_ATTR] as S on P.[PTNT_DK]=S.[PTNT_DK]
  Join [NATIONAL_ANALYTICS].[dbo].[DIM_PATIENT] DP ON P.EMPI = DP.[EMPI]

  Where P.[PRGM_STS] = 'Closed' and P.[TNT_MKT_BK]='CA'