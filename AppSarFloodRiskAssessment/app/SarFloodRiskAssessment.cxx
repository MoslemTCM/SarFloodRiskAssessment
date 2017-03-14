/*========================================================================= */

// CRIM : Centre de recherche en informatique de Montréal

// Équipe Télédetection pour les catastrophes majeures (TCM).

// Programme : Un module pour l'extraction de l'etendue des inondations en se basant sur des images radar.

// Auteur : Moslem Ouled Sghaier

// Version : 3.0

/*========================================================================= */

#include "otbImage.h"
#include "otbImageFileReader.h"
#include "otbImageFileWriter.h"
#include "itkCastImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkCannyEdgeDetectionImageFilter.h"
#include "otbSFSTexturesImageFilter.h"
#include "itkGrayscaleDilateImageFilter.h"
#include <itkGrayscaleMorphologicalOpeningImageFilter.h>
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkFlatStructuringElement.h"

#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include <iostream>
#include "itkVector.h"
#include "itkBresenhamLine.h"
#include <vector>
#include <math.h>
#include <ctime>

#include "itkImageDuplicator.h"

#include "itkFlipImageFilter.h"
#include "itkFixedArray.h"

#include <cstdlib>
#include <stdlib.h>
#include <math.h>

#include <conio.h>
#include <stdio.h>

// Seuillage
#include "itkBinaryThresholdImageFilter.h"

// inverser une image
#include <itkInvertIntensityImageFilter.h>

//Bresenham
#include "itkBresenhamLine.h"

#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkImageRegistrationMethod.h"

// Image  to LabelMap and LabelImage
#include "itkImageRegionIterator.h"
#include "itkBinaryImageToShapeLabelMapFilter.h"
#include "itkLabelMapToLabelImageFilter.h"
#include <itkLabelMapToBinaryImageFilter.h>
#include "otbWrapperApplication.h" 
#include "otbWrapperApplicationRegistry.h"
#include "otbWrapperApplicationFactory.h"
#include "otbWrapperTags.h"
#include "itkBinaryMorphologicalClosingImageFilter.h"

#include "itkConnectedComponentImageFilter.h"
#include "otbPersistentVectorizationImageFilter.h"
#include "otbVectorDataProjectionFilter.h"

#include "itkGrayscaleMorphologicalClosingImageFilter.h"
// Je vais essayer d'utiliser kmeans pour séparer l'image en deux classes différentes

// un duplicateur d'image
#include "itkImageDuplicator.h"

#include"itkImageAdaptor.h"

//Utils
#include "itksys/SystemTools.hxx"
#include "itkListSample.h"

// Elevation handler
#include "otbWrapperElevationParametersHandler.h"

// Image thining
#include "itkBinaryThinningImageFilter.h"

// Une image vecteur à une image
#include "itkVectorImageToImageAdaptor.h"

// ceci sera utile pour les données vectorielles
#include "otbVectorData.h" 
#include "otbVectorDataFileReader.h" 
#include "otbVectorDataFileWriter.h"

#include "itkPreOrderTreeIterator.h" 
#include "otbObjectList.h" 
#include "otbPolygon.h"
#include <otbVectorDataTransformFilter.h>

#include "itkTriangleThresholdImageFilter.h"

#include "itkResampleImageFilter.h"
#include "itkIdentityTransform.h"

// Liste des images qui seront utilisées comme entrées
#include <otbImageList.h>

// Application de la gaussienne
#include "itkDiscreteGaussianImageFilter.h"

// SOM map (Kohonan map)

#include "itkEuclideanDistanceMetric.h"

 
#include "itkComposeImageFilter.h"
#include "otbImageToVectorImageCastFilter.h"
#include "itkScalarImageKmeansImageFilter.h"
#include "itkImageKmeansModelEstimator.h"

// Classification bayesienne

#include "itkBayesianClassifierImageFilter.h"
#include "itkGradientAnisotropicDiffusionImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"

// Kmeans estimator

#include "itkImageKmeansModelEstimator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageToListSampleAdaptor.h"
#include "itkDistanceToCentroidMembershipFunction.h"
#include "itkSampleClassifierFilter.h"
#include "itkMinimumDecisionRule.h"

#include "itkKdTree.h" 
#include "itkWeightedCentroidKdTreeGenerator.h"
#include "itkKdTreeBasedKmeansEstimator.h"
#include "itkMinimumDecisionRule.h" 
#include "itkSampleClassifierFilter.h"
#include "itkNormalVariateGenerator.h"

//using namespace std;

extern "C" {
	#include "pde_toolbox_bimage.h"
	#include "pde_toolbox_defs.h"
//	#include "pde_toolbox_LSTB.h"
//	#include "ImageMagickIO.h"
}

#include "pathopenclose.h"

#include "itkChangeInformationImageFilter.h"
#include "itkVersor.h"

// Application des ondelettes sur l'image

#include "otbWaveletFilterBank.h"
#include "otbWaveletGenerator.h"
#include "otbWaveletOperator.h"
#include "otbWaveletTransform.h"

#include <boost/numeric/ublas/matrix.hpp>
#include "boost/multi_array.hpp"
#include <boost/numeric/ublas/io.hpp>
#include <boost/algorithm/minmax.hpp>

// Utiliser OGR pour écrire des shapefile

#include "ogrsf_frmts.h"

// La theorie des fonctions de croyance

#include <boost/bft/mass.hpp>
#include <boost/bft/rule_conjunctive.hpp>

BOOST_BFT_DEFINE_CLASS(C1);
BOOST_BFT_DEFINE_CLASS(C2);
BOOST_BFT_DEFINE_CLASS(C3);
typedef boost::bft::fod<C1, C2, C3> fod3;
using namespace boost::bft;
rule_conjunctive rule;
typedef fod3 fod_t;

// Declaration des structures utilisées dans la fusion 

struct data_base
{
    std::string classe;
    float moyenne;
};

struct masse_str
{
    std::string classe;
	double masse;
	double ingorance;
	
};

namespace otb
{
namespace Wrapper
{

class SarFloodRiskAssessment : public Application
{

public:
	typedef SarFloodRiskAssessment Self;
    typedef Application              Superclass;
    typedef itk::SmartPointer<Self>       Pointer;
    typedef itk::SmartPointer<const Self> ConstPointer;

#define PI 3.14159265


/** Standard macro */
    itkNewMacro(Self);
    itkTypeMacro(SarFloodRiskAssessment, otb::Application);

typedef unsigned char CharPixelType; // IO char
typedef float FloatPixelType; // IO int
typedef otb::Image<FloatPixelType, 2> FloatImageType;
typedef otb::Image<CharPixelType, 2> CharImageType;
typedef otb::ImageList<CharImageType> CharImageListType;
typedef itk::VectorImageToImageAdaptor<float,2> ImageAdaptorType;
CharImageType::Pointer ImageResultante;

private:

void DoInit()
{

SetName("SarFloodRiskAssessment"); // Nécessaire
SetDocName("SarFloodRiskAssessment");
SetDocLongDescription("Un module pour l'extraction de l'etendue des inondations en se basant sur des images radar");
SetDocLimitations("Les autres paramètres seront ajoutés plus tard");
SetDocAuthors("Moslem Ouled Sghaier");


AddParameter(ParameterType_InputImageList,"in", "The input list image");
SetParameterDescription("in", "The input list image");
AddParameter(ParameterType_OutputVectorData,"out", "The output vector");
SetParameterDescription("out","The output vector");
}

void DoUpdateParameters()
{
	// Nothing to do here : all parameters are independent
}

void DoExecute()
{

clock_t start, stop;
start = clock();

typedef unsigned char CharPixelType; // IO
const unsigned int Dimension = 2;
typedef otb::Image<CharPixelType, Dimension> CharImageType;

const int x = GetParameterImageList("in")->GetNthElement(0)->GetLargestPossibleRegion().GetSize()[0];
const int y =  GetParameterImageList("in")->GetNthElement(0)->GetLargestPossibleRegion().GetSize()[1];

// Utilistion de la structure de matrice pour le travail
using namespace boost::numeric::ublas;
matrix<std::vector<masse_str>> matrice_ondelettes (x, y);
matrix<std::vector<masse_str>> matrice_ondelettes1 (x, y);
matrix<mass<fod_t>> matrice_kmeans (x, y);
matrix<mass<fod_t>> matrice_kmeans1 (x, y);

FloatVectorImageListType* imageList = GetParameterImageList("in");
CharImageType::Pointer seuil;
CharImageType::Pointer seuil1;

   std::cout << "********************************** Debut du programme ******************************" << std::endl;

for (unsigned int imageId = 0; imageId+1 < imageList->Size(); ++imageId)
{
  FloatVectorImageType* image = imageList->GetNthElement(imageId); // Lecture de la première image
  FloatVectorImageType* image1 = imageList->GetNthElement(imageId+1);// Lecture de la deuxième image

  ImageAdaptorType::Pointer adaptor = ImageAdaptorType::New();
  ImageAdaptorType::Pointer adaptor1 = ImageAdaptorType::New();

  adaptor->SetExtractComponentIndex(0);
  adaptor1->SetExtractComponentIndex(0);

  adaptor->SetImage(image);
  adaptor->Update();

  adaptor1->SetImage(image1);
  adaptor1->Update();

  std::cout << "*************************** Correction radiométrique de l'image ********************" << std::endl;

  CharImageType::Pointer rescale  = Convert (adaptor);
  CharImageType::Pointer rescale1  = Convert (adaptor1);

  std::cout << "************************ Application de la mesure de texture SFS-SD ****************" << std::endl;

  CharImageType::Pointer SFSFilter = SFSTextures(rescale);
  CharImageType::Pointer SFSFilter1 = SFSTextures(rescale1);
  //affiche (SFSFilter, SFSFilter1);
  
  std::cout << "****************************** Filtrage des bords de l'image ***********************" << std::endl;
  
  SFSFilter = border(SFSFilter);
  SFSFilter1 = border(SFSFilter1);

  std::cout << "*********************** Application de l'ouverture morphologique *******************" << std::endl;

  unsigned int L=0;
  unsigned int L1=0;
  if (imageId == 0){
  L = 100; // La  valeur pour la premiere image seulement
  L1 = 50;
  }
  else{
  L = 50; // La  valeur pour le reste image seulement
  L1 = 50;
  }

  CharImageType::Pointer openingFilter = PathOpening(SFSFilter,L);
  CharImageType::Pointer openingFilter1 = PathOpening(SFSFilter1,L1);
  affiche (openingFilter, openingFilter1);
  
  std::cout << "******* Estimation des fonctions de masse en utilisant l'analyse multiéchelle de la texture **********" << std::endl;

  matrice_ondelettes = estimation_multiechelle_texture(openingFilter);
  matrice_ondelettes1 = estimation_multiechelle_texture(openingFilter1);

  std::cout << "************ Estimation des fonctions de masse en utilisant le kpp Évidentiel ***********" << std::endl;

  matrice_kmeans = estimation_Kmeans(openingFilter);
  matrice_kmeans1 = estimation_Kmeans(openingFilter1);
  
  //for (std::size_t i = 0; i < fod_t::powerset_size; ++i) {

  std::cout << "******************** Fusion des fonctions de masse *********************************" << std::endl;

  typedef itk::ImageDuplicator<CharImageType> ImageDuplicatorType;
  ImageDuplicatorType::Pointer duplicator = ImageDuplicatorType::New();
  duplicator->SetInputImage(openingFilter);
  duplicator->Update();
  ImageResultante = duplicator->GetOutput();
  ImageResultante = fusion_total (matrice_ondelettes, matrice_ondelettes1, matrice_kmeans, matrice_kmeans1, ImageResultante);
  ogr_save_shapefile(ImageResultante, imageId);
  std::cout << "*********************** Seuillage de l'image avant affichage ***********************" << std::endl;

  seuil = Seuillage(openingFilter);
  seuil1 = Seuillage(openingFilter1);
  
  std::cout << " J'ai parcouru :" << imageId <<  "et" << imageId+1 << " images de " << imageList->Size() << std::endl;
}
 
  VectorDataType::Pointer vectorDataProjection = VectorDataType::New();

  vectorDataProjection = Affichage(ImageResultante);
 
  SetParameterOutputVectorData("out",vectorDataProjection);

  stop = clock();

  std::cout << "CPU time elapsed:" << ((double)stop-start)/CLOCKS_PER_SEC << std::endl;

  std::cout << "********************************** Fin du programme ******************************" << std::endl;

}

void ogr_save_shapefile (CharImageType::Pointer ImageResultante, int imageId)
{   
	char vecteur[50];  
	sprintf(vecteur, "amouna %d.shp",imageId);

	VectorDataType::Pointer vectorDataProjection = VectorDataType::New();
	vectorDataProjection = Affichage(ImageResultante);

	typedef otb::VectorDataFileWriter<VectorDataType> WriterType;
	WriterType::Pointer writer = WriterType::New(); 
    writer->SetInput(vectorDataProjection); 
    writer->SetFileName(vecteur); 
    writer->Update();

}

CharImageType::Pointer Convert (ImageAdaptorType* input)
{
    typedef itk::RescaleIntensityImageFilter<ImageAdaptorType,CharImageType> RescaleFilter0;
    RescaleFilter0::Pointer rescale1 = RescaleFilter0::New();
    rescale1->SetInput(input);
	rescale1->Update();

	return (rescale1->GetOutput());
}


CharImageType::Pointer SFSTextures (CharImageType::Pointer image)
{
   typedef otb::SFSTexturesImageFilter<CharImageType, CharImageType> SFSFilterType;
   SFSFilterType::Pointer SFSFilter1 = SFSFilterType::New();
   SFSFilter1->SetSpectralThreshold(8);
   SFSFilter1->SetSpatialThreshold(100); 
   SFSFilter1->SetNumberOfDirections(20); 
   SFSFilter1->SetRatioMaxConsiderationNumber(5); 
   SFSFilter1->SetAlpha(1.00);
   SFSFilter1->SetInput(image);
   SFSFilter1->Update();

	return (SFSFilter1->GetOutput());
  } 

CharImageType::Pointer border (CharImageType::Pointer SFSFilter1)

{
  unsigned int x= SFSFilter1->GetLargestPossibleRegion().GetSize()[0];
  unsigned int y= SFSFilter1->GetLargestPossibleRegion().GetSize()[1];

  for (unsigned int i=0 ; i< 2;i++ )
  for (unsigned int j=0 ; j< y;j++)
  {{
  CharImageType::IndexType pixelIndex; 
  pixelIndex[0] = i;   // x position 
  pixelIndex[1] = j;   // y position
  SFSFilter1->SetPixel(pixelIndex,0);}}

  for (unsigned int i=0 ; i< x;i++ )
  for (unsigned int j=0 ; j< 2;j++)
  {{
  CharImageType::IndexType pixelIndex; 
  pixelIndex[0] = i;   // x position 
  pixelIndex[1] = j;   // y position
  SFSFilter1->SetPixel(pixelIndex,0);}}

  for (unsigned int i=x-2 ; i< x;i++ )
  for (unsigned int j=0 ; j< y;j++)
  {{
  CharImageType::IndexType pixelIndex; 
  pixelIndex[0] = i;   // x position 
  pixelIndex[1] = j;   // y position
  SFSFilter1->SetPixel(pixelIndex,0);}}

  for (unsigned int i=0 ; i< x;i++ )
  for (unsigned int j=y-2 ; j< y;j++)
  {{
  CharImageType::IndexType pixelIndex; 
  pixelIndex[0] = i;   // x position 
  pixelIndex[1] = j;   // y position
  SFSFilter1->SetPixel(pixelIndex,0);}}

  return (SFSFilter1);
}

CharImageType::Pointer Closing (CharImageType::Pointer input)
  
 {
  
  typedef itk::BinaryBallStructuringElement< CharPixelType,2 > StructuringElementType;
  typedef itk::GrayscaleMorphologicalClosingImageFilter< CharImageType, CharImageType, StructuringElementType >  ClosingFilterType;

  ClosingFilterType::Pointer closingFilter = ClosingFilterType::New();

  closingFilter->SetInput(input);
  StructuringElementType elementRadius;
  elementRadius.SetRadius(2);
  elementRadius.CreateStructuringElement();
  closingFilter->SetKernel(elementRadius);
  closingFilter->GetSafeBorder();
  closingFilter->Update();
  return (closingFilter->GetOutput());
 }

CharImageType::Pointer PathOpening (CharImageType::Pointer openingFilter, int L)
 {
    int   i, K;

    K=1;

  unsigned int nx = openingFilter->GetLargestPossibleRegion().GetSize()[0];//input_bimage->dim->buf[0];
  unsigned int ny = openingFilter->GetLargestPossibleRegion().GetSize()[1];;//input_bimage->dim->buf[1];
  unsigned int num_pixels = nx*ny;

  std::cout << num_pixels << std::endl;

  PATHOPEN_PIX_TYPE * input_image = new PATHOPEN_PIX_TYPE[nx * ny];
  PATHOPEN_PIX_TYPE * output_image = new PATHOPEN_PIX_TYPE[nx * ny];
  
  // Convert intermediate float to PATHOPEN_PIX_TYPE (unsigned char)
	for (i = 0; i < num_pixels; ++i) {
		CharImageType::IndexType index = {i % nx, i / nx};
		input_image[i] = openingFilter->GetPixel(index);//static_cast<PATHOPEN_PIX_TYPE>(input_bimage->buf[i]);
	}

	std::cout << "Calling pathopen ()" << std::endl;
        
	pathopen(
            input_image, // The input image //
            nx, ny,	 // Image dimensions //
            L,		 // The threshold line length //
            K,		 // The maximum number of gaps in the path //
            output_image // Output image //
            ); 

	for (i = 0; i < num_pixels; ++i) {
		CharImageType::IndexType index = {i % nx, i / nx};
		openingFilter->SetPixel(index,  static_cast<CharImageType::PixelType>(output_image[i]));
	}

	return (openingFilter);
 }

boost::numeric::ublas::matrix<mass<fod_t>> estimation_Kmeans (CharImageType::Pointer input)

{

    // La matrice estimation
    boost::numeric::ublas::matrix<std::vector<masse_str>> matrice_kmeans (input->GetLargestPossibleRegion().GetSize()[0], input->GetLargestPossibleRegion().GetSize()[1]);
	boost::numeric::ublas::matrix<mass<fod_t>> matrice_kmeans_resultat (input->GetLargestPossibleRegion().GetSize()[0], input->GetLargestPossibleRegion().GetSize()[1]);
	boost::numeric::ublas::matrix<int> matrice_temp (input->GetLargestPossibleRegion().GetSize()[0], input->GetLargestPossibleRegion().GetSize()[1]);

   typedef itk::Statistics::ImageToListSampleAdaptor<CharImageType> AdaptorType;
   AdaptorType::Pointer adaptor = AdaptorType::New();
   adaptor->SetImage(input);
   adaptor->Update();

   // Create the K-d tree structure
  typedef itk::Statistics::WeightedCentroidKdTreeGenerator<AdaptorType> TreeGeneratorType;

  TreeGeneratorType::Pointer treeGenerator = TreeGeneratorType::New();

  treeGenerator->SetSample(adaptor);
  treeGenerator->SetBucketSize(16);
  treeGenerator->Update();

  typedef TreeGeneratorType::KdTreeType                         TreeType;
  typedef itk::Statistics::KdTreeBasedKmeansEstimator<TreeType> EstimatorType;

  EstimatorType::Pointer estimator = EstimatorType::New();

  const unsigned int numberOfClasses = 3;

  EstimatorType::ParametersType initialMeans(numberOfClasses);

  initialMeans[0] = 25.0;
  initialMeans[1] = 125.0;
  initialMeans[2] = 255.0;

  estimator->SetParameters(initialMeans);
  
  estimator->SetKdTree(treeGenerator->GetOutput());
  estimator->SetMaximumIteration(400);
  estimator->SetCentroidPositionChangesThreshold(0.0);
  estimator->StartOptimization();

  EstimatorType::ParametersType estimatedMeans = estimator->GetParameters();

  for (unsigned int i = 0; i < numberOfClasses; ++i)
    {
    std::cout << "cluster[" << i << "] " << std::endl;
    std::cout << "    estimated mean : " << estimatedMeans[i] << std::endl;
    }
   
  // matrice_temp = detemrine_classe_dappartenance (input, numberOfClasses, estimatedMeans);

  // Replissage de la base de données
  std::vector<data_base> data;

  data_base ele1; // non homogene non grande taille
  ele1.classe = "not homogene not grand";
  ele1.moyenne = estimatedMeans[0];

  data_base ele2; // non homogene non grande taille
  ele2.classe = "homogene not grand";
  ele2.moyenne = estimatedMeans[1];

  data_base ele3; // non homogene non grande taille
  ele3.classe = "homogene grand";
  ele3.moyenne = estimatedMeans[2];

  data.push_back(ele1);
  data.push_back(ele2);
  data.push_back(ele3);
  //////////////////////////////////////

  int nombre_classe=3;

  // EK_NN à partir de la matrice temp
  matrice_kmeans_resultat = Ek_nn(data, input, nombre_classe);

  return matrice_kmeans_resultat;

}

int trouve (int min, std::vector<int> ele )

 {
	 for (int p= 0 ; p < ele.size() ; p++ )
	 
	 {
	   if (ele.at(p) == min )
	   return 1;
	 
	 }

	 return 0;
 }

boost::numeric::ublas::matrix<mass<fod_t>> combine_masses ( boost::numeric::ublas::matrix<std::vector<masse_str>> matrice_temp )
{  
	boost::numeric::ublas::matrix<mass<fod_t>> matrice_temp_resultat (matrice_temp.size1(), matrice_temp.size2());

	for (int i = 0; i < matrice_temp.size1() ; i++)
	{
	   for (int j = 0; j < matrice_temp.size2() ; j++ )
	   { 
	 const mass<fod_t>::container_type ma1 = {matrice_temp(i,j).at(0).masse, 0, 0, matrice_temp(i,j).at(0).ingorance};
     const mass<fod_t> m1(ma1);
     const mass<fod_t>::container_type ma2 = {0, matrice_temp(i,j).at(1).masse, 0, matrice_temp(i,j).at(1).ingorance};
     const mass<fod_t> m2(ma2);
	 const mass<fod_t>::container_type ma3 = {0, 0, matrice_temp(i,j).at(2).masse, matrice_temp(i,j).at(2).ingorance};
     const mass<fod_t> m3(ma3);

     mass<fod_t> m12 = m1.apply(rule, m2);
	 m12 = m12.apply(rule, m3);

	 matrice_temp_resultat(i,j) = m12; 
	   }
	}
	return matrice_temp_resultat;
}

boost::numeric::ublas::matrix<mass<fod_t>> Ek_nn(std::vector<data_base> data, CharImageType::Pointer input, int nombre_classe)

{
std::vector<std::vector<masse_str>> fusion_final;

	// On determine la classe de chaque région
boost::numeric::ublas::matrix<std::vector<masse_str>> matrice_temp (input->GetLargestPossibleRegion().GetSize()[0], input->GetLargestPossibleRegion().GetSize()[1]);
boost::numeric::ublas::matrix<mass<fod_t>> matrice_temp_resultat (input->GetLargestPossibleRegion().GetSize()[0], input->GetLargestPossibleRegion().GetSize()[1]);

for (int i=0; i < input->GetLargestPossibleRegion().GetSize()[0] ; i++)
{
 for (int j=0; j < input->GetLargestPossibleRegion().GetSize()[1] ; j++)
     {  
       CharImageType::IndexType pixelIndex;
       pixelIndex[0]=i; pixelIndex[1]=j;
       CharImageType::PixelType pixelValue = input->GetPixel(pixelIndex);

       for (int k=0; k < data.size() ; k++)
	       {
	        masse_str* m= new masse_str();
	        m->classe = data.at(k).classe;
			double resultat = (double) 0.95 / ( std::pow( ( (float) data.at(k).moyenne - (int) pixelValue ) ,2) + 1 );
	        m->masse = resultat;
	        m->ingorance =(double) 1- m->masse;
	        matrice_temp(i,j).push_back(*m);

	        free(m);
           }
     }
}
  matrice_temp_resultat = combine_masses (matrice_temp);
  
  return matrice_temp_resultat ;

}

boost::numeric::ublas::matrix<int>  detemrine_classe_dappartenance (CharImageType::Pointer input, const unsigned int numberOfClasses, itk::Statistics::KdTreeBasedKmeansEstimator< itk::Statistics::WeightedCentroidKdTreeGenerator<itk::Statistics::ImageToListSampleAdaptor<CharImageType>>::KdTreeType>::ParametersType estimatedMeans)

{
	boost::numeric::ublas::matrix<int> matrice_temp (input->GetLargestPossibleRegion().GetSize()[0], input->GetLargestPossibleRegion().GetSize()[1]);

	for (unsigned int i= 0; i < matrice_temp.size1(); i++ )
	{
	   for (unsigned int j=0; j < matrice_temp.size2(); j++)
	   {
		   CharImageType::IndexType pixelIndex;
		   pixelIndex[0]=i; pixelIndex[1]=j;
		   CharImageType::PixelType pixelValue = input->GetPixel(pixelIndex);

		   if ( ( std::abs((int)pixelValue - estimatedMeans[0]) < std::abs((int)pixelValue - estimatedMeans[1]) ) && ( std::abs((int)pixelValue - estimatedMeans[0]) < std::abs((int)pixelValue - estimatedMeans[2]) ) )
	       matrice_temp(i,j) = 0;
		   else if ( ( std::abs((int)pixelValue - estimatedMeans[1]) < std::abs((int)pixelValue - estimatedMeans[2]) ) && ( std::abs((int)pixelValue - estimatedMeans[1]) < std::abs((int)pixelValue - estimatedMeans[0]) ) )
           matrice_temp(i,j) = 1;
		   else if ( ( std::abs((int)pixelValue - estimatedMeans[2]) < std::abs((int)pixelValue - estimatedMeans[1]) ) && ( std::abs((int)pixelValue - estimatedMeans[2]) < std::abs((int)pixelValue - estimatedMeans[0]) ) )
		   matrice_temp(i,j) = 2;
	   }
	}
	return matrice_temp ; 

}


boost::numeric::ublas::matrix<std::vector<masse_str>> estimation_multiechelle_texture (CharImageType::Pointer input)
  
 { 
    // Je vais juste afficher l'image d'entrée pour s'assurer que tout va bien
	
	typedef float FloatPixelType; // IO
    const unsigned int Dimension = 2;
    typedef otb::Image<FloatPixelType, Dimension> FloatImageType;

	const unsigned int requestedLevel = 4;
    float frac_estimation = 1 / (requestedLevel + 1);

	std::string classe1 = "not homogene not grand";
	std::string classe2 = "homogene not grand";
	std::string classe3 = "homogene grand";

	// La matrice estimation

	boost::numeric::ublas::matrix<std::vector<masse_str>> matrice_ondelettes (input->GetLargestPossibleRegion().GetSize()[0], input->GetLargestPossibleRegion().GetSize()[1]);
	boost::numeric::ublas::matrix<double> matrice_temp (input->GetLargestPossibleRegion().GetSize()[0], input->GetLargestPossibleRegion().GetSize()[1]);

	// Application de l'ouverture morphologique (multiéchelle)

	for(unsigned int loop = 50; loop <= 300 ; loop=loop+50)
	{
		 
	typedef itk::ImageDuplicator<CharImageType> ImageDuplicatorType;
	ImageDuplicatorType::Pointer duplicator = ImageDuplicatorType::New();
	duplicator->SetInputImage(input);
	duplicator->Update();
	CharImageType::Pointer temp = duplicator->GetOutput();

	CharImageType::Pointer output = PathOpening (temp, loop);

	// Je rempli la matrice d'estimation temporelle
    
	for ( unsigned int i = 0; i < matrice_temp.size1() ; i++)
		{
    for ( unsigned int j = 0; j < matrice_temp.size2() ; j++)
    	{

    CharImageType::IndexType pixelIndex;
	pixelIndex[0] = i;   // x position
    pixelIndex[1] = j;   // y position
	CharImageType::PixelType pixelValue = output->GetPixel(pixelIndex);

	matrice_temp(i,j) = matrice_temp(i,j) + pixelValue ;
	matrice_ondelettes(i,j).clear();   
	   }}
	}

	double h = maximum(matrice_temp);

	for (int i = 0; i < matrice_temp.size1(); i++){
		  for (int j = 0; j < matrice_temp.size2(); j++)
		  {
          	masse_str* m= new masse_str();
	        m->classe = classe3;
	        m->masse =(double) matrice_temp(i,j) / h;
			m->ingorance =(double) 1 -  m->masse;

			matrice_ondelettes(i,j).push_back(*m);
			free(m);

          }}

	return matrice_ondelettes ;

 }

  double maximum ( boost::numeric::ublas::matrix<double> matrice_temp )
   
  {   double max = 0;

  for (unsigned int i = 0; i < matrice_temp.size1() ; i++){
		  for (unsigned int j = 0; j < matrice_temp.size2() ; j++)
		  {    
			   if (matrice_temp(i,j) > max)
			   max = matrice_temp(i,j);
		  }}
    return max;
  }

   void affiche (CharImageType::Pointer rescale, CharImageType::Pointer rescale1)
   {
    // juste pour l'affichage 
	char output_name124[50];  
	char output_name125[50]; 

    typedef otb::ImageFileWriter<CharImageType> WriterType124;
	typedef otb::ImageFileWriter<CharImageType> WriterType125;
    WriterType124::Pointer writer124 = WriterType124::New();
	WriterType125::Pointer writer125 = WriterType125::New();

    sprintf(output_name124, "1.tif");
	sprintf(output_name125, "2.tif");

	writer124->SetFileName(output_name124);
	writer124->SetInput(rescale);
    writer124->Update();

    writer125->SetFileName(output_name125);
	writer125->SetInput(rescale1);
    writer125->Update();
   }
	    
   CharImageType::Pointer fusion_total (boost::numeric::ublas::matrix<std::vector<masse_str>> matrice_ondelettes, 
   boost::numeric::ublas::matrix<std::vector<masse_str>> matrice_ondelettes1, boost::numeric::ublas::matrix<mass<fod_t>> matrice_kmeans,
   boost::numeric::ublas::matrix<mass<fod_t>> matrice_kmeans1, CharImageType::Pointer ImageResultante)
  {
	boost::numeric::ublas::matrix<mass<fod_t>> mat_resu (matrice_ondelettes.size1(), matrice_ondelettes.size2());
	boost::numeric::ublas::matrix<mass<fod_t>> mat_resu1 (matrice_ondelettes1.size1(), matrice_ondelettes1.size2());
	boost::numeric::ublas::matrix<std::vector<int>> final (matrice_ondelettes1.size1(), matrice_ondelettes1.size2());
    
	// La fusion pour l'image 1
	for (unsigned int i = 0; i < matrice_ondelettes.size1(); i++){
		  for (unsigned int j = 0; j < matrice_ondelettes.size2(); j++)
		  {
			const mass<fod_t>::container_type ma2 = {0, 0, matrice_ondelettes(i,j).at(0).masse,  matrice_ondelettes(i,j).at(0).ingorance};
            const mass<fod_t> m2(ma2);	

			mat_resu(i,j) = matrice_kmeans(i,j).apply(rule, m2);
		  }}
       matrice_kmeans.resize(0,0,false);
      // La fusion pour l'image 2
	for (unsigned int i1 = 0; i1 < matrice_ondelettes1.size1(); i1++) {
		  for (unsigned int j1 = 0; j1 < matrice_ondelettes1.size2(); j1++)
		 {
			const mass<fod_t>::container_type ma21 = {0, 0, matrice_ondelettes1(i1,j1).at(0).masse,  matrice_ondelettes1(i1,j1).at(0).ingorance};
            const mass<fod_t> m21(ma21);	

			mat_resu1(i1,j1) = matrice_kmeans1(i1,j1).apply(rule, m21);
		  }}
	matrice_kmeans1.resize(0,0,false);
   
	// Fusion multitemporelle 
	for (unsigned int i2 = 0; i2 < mat_resu.size1(); i2++) {
		  for (unsigned int j2 = 0; j2 < mat_resu.size2(); j2++)
		 {
			 if ( (mat_resu(i2,j2)[0] > mat_resu(i2,j2)[1]) && (mat_resu(i2,j2)[0] > mat_resu(i2,j2)[2])  )
				 final(i2,j2).push_back(1);
			 else if ((mat_resu(i2,j2)[1] > mat_resu(i2,j2)[2]) && (mat_resu(i2,j2)[1] > mat_resu(i2,j2)[0]))
				 final(i2,j2).push_back(2);
			 else if ((mat_resu(i2,j2)[2] > mat_resu(i2,j2)[1]) && (mat_resu(i2,j2)[2] > mat_resu(i2,j2)[0]))
				 final(i2,j2).push_back(3);

			 if ( (mat_resu1(i2,j2)[0] > mat_resu1(i2,j2)[1]) && (mat_resu1(i2,j2)[0] > mat_resu1(i2,j2)[2])  )
				 final(i2,j2).push_back(1);
			 else if ((mat_resu1(i2,j2)[1] > mat_resu1(i2,j2)[2]) && (mat_resu1(i2,j2)[1] > mat_resu1(i2,j2)[0]))
				 final(i2,j2).push_back(2);
			 else if ((mat_resu1(i2,j2)[2] > mat_resu1(i2,j2)[1]) && (mat_resu1(i2,j2)[2] > mat_resu1(i2,j2)[0]))
				 final(i2,j2).push_back(3);

		  }}
	   mat_resu.resize(0,0,false);   
	   mat_resu1.resize(0,0,false);

	   // Replissage de la matrice résultante
   for (int q = 0; q < final.size1(); q++)
    {
      for (int w = 0; w < final.size2(); w++) 
	  {    
        CharImageType::IndexType pixelIndex;
		pixelIndex[0] = q;   // x position 
        pixelIndex[1] = w;   // y position
		
		if ( (final(q,w).at(0) == 1) && (final(q,w).at(1) == 3) ) 
			ImageResultante->SetPixel(pixelIndex,255);
		else
	   		ImageResultante->SetPixel(pixelIndex,0);
	  }
    }

    return ImageResultante;
  }

  CharImageType::Pointer Seuillage (CharImageType::Pointer openingFilter)
 {
  typedef itk::BinaryThresholdImageFilter<CharImageType, CharImageType>  FilterType1;
  FilterType1::Pointer filter1 = FilterType1::New();
  filter1->SetInput(openingFilter);
  filter1->SetLowerThreshold(40);
  filter1->Update();

  return (filter1->GetOutput());
 }

 VectorDataType::Pointer Affichage(CharImageType::Pointer seuil)
  {

  typedef otb::Polygon<double>             PolygonType;
  typedef PolygonType::Pointer             PolygonPointerType;
  typedef PolygonType::ContinuousIndexType PolygonIndexType;
  typedef otb::ObjectList<PolygonType>     PolygonListType;
  typedef PolygonListType::Pointer         PolygonListPointerType;
  typedef unsigned long LabelPixelType;
  typedef otb::Image<LabelPixelType, 2>  LabeledImageType;

  typedef itk::ConnectedComponentImageFilter<CharImageType,LabeledImageType> ConnectedFilterType;
  typedef otb::PersistentVectorizationImageFilter<LabeledImageType,PolygonType> PersistentVectorizationFilterType;

  ConnectedFilterType::Pointer connectedFilter = ConnectedFilterType::New();
  connectedFilter->SetInput(seuil);

  //Perform vectorization in a persistent way
    PersistentVectorizationFilterType::Pointer persistentVectorization = PersistentVectorizationFilterType::New();
    persistentVectorization->Reset();
    persistentVectorization->SetInput(connectedFilter->GetOutput());
    try
      {
      persistentVectorization->Update();
      }
    catch (itk::ExceptionObject& err)
      {
      std::cout << "\nExceptionObject caught !" << std::endl;
      std::cout << err << std::endl;
      }

	PolygonListPointerType OutputPolyList = persistentVectorization->GetPathList();
	//Display results
    std::cout << "nb objects found = " << OutputPolyList->Size() << std::endl;
	
	VectorDataType::Pointer outVectorData = VectorDataType::New(); 
    typedef VectorDataType::DataNodeType DataNodeType;
	typedef VectorDataType::DataTreeType            DataTreeType; 
    typedef itk::PreOrderTreeIterator<DataTreeType> TreeIteratorType;

	DataNodeType::Pointer document = DataNodeType::New(); 
    document->SetNodeType(otb::DOCUMENT); 
    document->SetNodeId("polygon"); 
    DataNodeType::Pointer folder = DataNodeType::New(); 
    folder->SetNodeType(otb::FOLDER); 
    DataNodeType::Pointer multiPolygon = DataNodeType::New(); 
    multiPolygon->SetNodeType(otb::FEATURE_MULTIPOLYGON);

	DataTreeType::Pointer tree = outVectorData->GetDataTree(); 
    DataNodeType::Pointer root = tree->GetRoot()->Get(); 
 
    tree->Add(document, root); 
    tree->Add(folder, document); 
    tree->Add(multiPolygon, folder);

	for (PolygonListType::Iterator pit = OutputPolyList->Begin(); 
       pit != OutputPolyList->End(); ++pit) 
    { 
		if (pit.Get()->GetSurface()  >  30){
    DataNodeType::Pointer newPolygon = DataNodeType::New(); 
    newPolygon->SetPolygonExteriorRing(pit.Get()); 
    tree->Add(newPolygon, multiPolygon); 
		}
    }

	typedef itk::AffineTransform<VectorDataType::PrecisionType, 2> TransformType;
	typedef otb::VectorDataTransformFilter <VectorDataType,VectorDataType > VectorDataFilterType;
    VectorDataFilterType:: Pointer vectorDataProjection = VectorDataFilterType:: New ();

	TransformType::ParametersType params;
    params.SetSize(6);
    params[0] = GetParameterImageList("in")->GetNthElement(0)->GetSpacing()[0];
    params[1] = 0;
    params[2] = 0;
    params[3] = GetParameterImageList("in")->GetNthElement(0)->GetSpacing()[1];
    params[4] = GetParameterImageList("in")->GetNthElement(0)->GetOrigin()[0];
    params[5] = GetParameterImageList("in")->GetNthElement(0)->GetOrigin()[1];
	TransformType::Pointer transform = TransformType::New();
    transform->SetParameters(params);

	vectorDataProjection->SetTransform(transform);
	vectorDataProjection->SetInput(outVectorData);
	vectorDataProjection->Update();

	std::cout << "OK" << std::endl;
	std::cout << "L'origine1:" << vectorDataProjection->GetOutput()->GetOrigin() << GetParameterImageList("in")->GetNthElement(0)->GetOrigin() << std::endl;
    std::cout << "L'espacement1:" << vectorDataProjection->GetOutput()->GetSpacing() << GetParameterImageList("in")->GetNthElement(0)->GetSpacing() << std::endl << std::endl;
	std::cout << "La projection1:"  << vectorDataProjection->GetOutput()->GetProjectionRef() << GetParameterImageList("in")->GetNthElement(0)->GetProjectionRef() << std::endl << std::endl;

	return (vectorDataProjection->GetOutput());
  }

     };
}
   
}
OTB_APPLICATION_EXPORT(otb::Wrapper::SarFloodRiskAssessment);