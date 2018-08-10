SELECT PAF.[PTNT_ASES_DK]
      ,PAF.[PTNT_DK]
      ,PAF.[ASES_ID]
      ,PAF.[SRC_TBL]
      ,PAF.[ASES_NM]
      ,PAF.[ASES_DT]
      ,PAF.[ASES_SCOR]
      ,PAF.[TNT_BK]
      ,PAF.[TNT_MKT_BK]
      ,PAF.[SRC_CD]
      ,PAF.[CHK_SM]
	  ,PD.[EMPI]
	  --,DP.[DOB]
	  --,DP.[Sex]
  FROM [CIM_RPT].[dbo].[PTNT_ASES_FCT] AS PAF
  JOIN [CIM_RPT].[dbo].[PTNT_DIM] AS PD ON PAF.PTNT_DK = PD.PTNT_DK
  --JOIN [NATIONAL_ANALYTICS].[dbo].[DIM_PATIENT] DP ON PD.EMPI = DP.[EMPI]
  WHERE ASES_NM = 'LACE'
  AND PAF.SRC_CD = 'CIM'
  AND PAF.[TNT_MKT_BK] = 'CA'
  AND ASES_DT >= '2018-04-01'