set(DOCUMENTATION "Suivi de l'étendue des inondations à partir des images radars.")

otb_module(OTBAppSarFloodRiskAssessment

  DEPENDS
    OTBITK
    OTBApplicationEngine
	OTBGdalAdapters
    OTBApplicationEngine
    OTBImageBase
    OTBCommon
    OTBImageManipulation
	OTBTextures
	OTBEdge
	
	
  TEST_DEPENDS
    OTBTestKernel
    OTBCommandLine

  DESCRIPTION
    "${DOCUMENTATION}"
)
