#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<sstream>


#include "PossibleChar.h"
#include "Recognition.h"

///Minimalus plotas konturo, kad ji laikytume simboliu
const int MIN_CONTOUR_AREA = 850; //100

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;


class ContourWithData {
public:
	///Vektoriaus tipo kintamasis, laiko savyje konturus
	std::vector<cv::Point> ptContour;   
	///Rect tipo kintamasis, kuris turi savyje konturo apvadus
	cv::Rect boundingRect;   
	///Float tipo kintamasis, savyje laiko konturo plota
	float fltArea;                              

												
	/*!Metodas tikrinti ar konturas panasus i skaiciu,
	*
	*Paima kontura ir tikrina atitinka minimalu plota,
	* ar jis yra simbolio formos.
	* Jei praeina kriterijus grazina true, jei ne false
	*/
	bool checkIfContourIsValid(PossibleChar &possibleChar) {
		if (fltArea < MIN_CONTOUR_AREA) return false;
		else {
			if (possibleChar.boundingRect.height*2 < possibleChar.boundingRect.width)
			{
				return false;
			}
		}
	}

	/**
	*Metodas rikiuoja konturus pagal x koordinate is kaires i desine
	*/
	static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {      
		return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);                                                   
	}

};


int main() {
	std::vector<ContourWithData> allContoursWithData;          
	std::vector<ContourWithData> validContoursWithData;        

																

	cv::Mat matClassificationInts;      

	cv::FileStorage fsClassifications("classifications1.xml", cv::FileStorage::READ);        

	if (fsClassifications.isOpened() == false) {                                                    
		std::cout << "Klaida, nepavyko atidaryti klasifikaciju failo, isjungiama\n\n";    
		return(0);                                                                                  
	}
	fsClassifications["classifications"] >> matClassificationInts;      
	fsClassifications.release();                                        


	cv::Mat matTrainingImagesAsFlattenedFloats;   

	cv::FileStorage fsTrainingImages("images1.xml", cv::FileStorage::READ);       

	if (fsTrainingImages.isOpened() == false) {                                                
		std::cout << "Klaida, nepavyko atidaryti trainingo failo, isjungiama\n\n";        
		return(0);                                                                             
	}

	fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;//surasom i training vektoriu          
	fsTrainingImages.release();                                                

	cv::Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());            


	kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);

	cv::Mat matNumbers = cv::imread("im1.png");           
	
	if (matNumbers.empty()) {                                
		std::cout << "Klaida, nepavyksta nuskaityti nuotraukos\n\n";         
		return(0);                                                  
	}

	cv::Mat matGrayscale;           
	cv::Mat matBlurred;             
	cv::Mat matThresh;              
	cv::Mat matThreshCopy;          
	cv::Mat matThreshCopy1; //open funkcijai
	cv::Mat matThreshCopy2; //close funkcijai

	cv::cvtColor(matNumbers, matGrayscale, CV_BGR2GRAY); //i grayscale


	cv::GaussianBlur(matGrayscale, //blurras
		matBlurred,                
		cv::Size(5, 5),            
		0);                        

	
	cv::adaptiveThreshold(matBlurred, //thresholdas                     
		matThresh,                            
		255,                                 
		cv::ADAPTIVE_THRESH_GAUSSIAN_C,       
		cv::THRESH_BINARY_INV,                
		11,                                   
		2);                                   


	matThreshCopy = matThresh.clone();
	matThreshCopy1 = matThresh.clone(); //open
	matThreshCopy2 = matThresh.clone(); //close

	cv::morphologyEx(matThreshCopy, //Sumazinam noise
		matThreshCopy1,
		cv::MORPH_OPEN,
		cv::Mat(),
		cv::Point(-1, -1),
		1);

	//cv::imshow("bla", imgThreshCopy2);

	cv::morphologyEx(matThreshCopy1, //Sujungiam per tarpus
		matThreshCopy2,
		cv::MORPH_CLOSE,
		cv::Mat(),
		cv::Point(-1, -1),
		3);

	std::vector<std::vector<cv::Point> > ptContours;  //Vektorius konturams
	std::vector<cv::Vec4i> v4iHierarchy;   

	cv::findContours(matThreshCopy2, //Ieskom konturu           
		ptContours,                            
		v4iHierarchy,                         
		cv::RETR_EXTERNAL,                     
		cv::CHAIN_APPROX_SIMPLE);              

	///
	///main metodas atlieka didziaja dali programos funkciju, 
	///grayscale, threshold gavimas, morfologines funkcijos, 
	///contoursWithData kvietimas ir simboliu ieskojimas, gautu simboliu isvedimas
	///
	for (int i = 0; i < ptContours.size(); i++) {  
		ContourWithData contourWithData;                                                  
		contourWithData.ptContour = ptContours[i];   //Jei konturas tinka pridedam prie contour with data
		contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);        
		contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);      
		allContoursWithData.push_back(contourWithData);     
	}



	for (int i = 0; i < allContoursWithData.size(); i++) {   
		PossibleChar possibleChar(ptContours[i]);
		if (allContoursWithData[i].checkIfContourIsValid(possibleChar)) {  //Patikrinam ar gali buti simbolis
			validContoursWithData.push_back(allContoursWithData[i]);       //Jei taip pridedam prie sekos
		}
	}
	//Rikiuojam
	std::sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortByBoundingRectXPosition);

	std::string strFinalString; //outputas

	for (int i = 0; i < validContoursWithData.size(); i++) {      

																		
		cv::rectangle(matNumbers,                          
			validContoursWithData[i].boundingRect,       
			cv::Scalar(0, 255, 0),                        
			1);                                          

		cv::Mat matROI = matThresh(validContoursWithData[i].boundingRect);          

		cv::Mat matROIResized;
		cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));    

		cv::Mat matROIFloat;
		matROIResized.convertTo(matROIFloat, CV_32FC1);            

		cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);

		cv::Mat matCurrentChar(0, 0, CV_32F);

		kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     

		float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

		strFinalString = strFinalString + char(int(fltCurrentChar));  //pridedam prie outputo
	}

	std::cout << "\n\n" << "Rasta skaiciu seka = " << strFinalString << "\n\n";    

	cv::imshow("Skaiciai", matNumbers);     

	cv::waitKey(0);                                         

	return(0);
}

