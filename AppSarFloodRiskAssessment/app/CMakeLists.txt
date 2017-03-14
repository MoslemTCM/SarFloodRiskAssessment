set(OTBAppSarFloodRiskAssessment_LINK_LIBS
  ${OTBTextures_LIBRARIES}
  ${OTBImageBase_LIBRARIES}
  ${OTBObjectList_LIBRARIES}
  ${OTBApplicationEngine_LIBRARIES}
  ${OTBImageManipulation_LIBRARIES}
  ${OTBGDAL_LIBRARIES}
)

set(OTBlib_c
   bimage.c
   path_queue.cxx
   path_support.c
   pathopen.cxx
   otbWaveletGenerator.cxx
   otbWaveletInverseTransformNew.cxx
   otbWaveletTransform.cxx
   otbWaveletTransformNew.cxx
  )


OTB_CREATE_APPLICATION(NAME SarFloodRiskAssessment
                       SOURCES SarFloodRiskAssessment.cxx ${OTBlib_c}
					   LINK_LIBRARIES ${${otb-module}_LIBRARIES})
					   
                      
#add_library(otbapp_SarFloodRiskAssessment ${OTBlib_c})