#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<vector>

const int MIN_CONTOUR_AREA = 850;

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

///////////////////////////////////////////////////////////////////////////////////////////////////
int main() {

	cv::Mat trainingImage;
	cv::Mat trainingImageH;
	cv::Mat imgGrayscale;
	cv::Mat imgBlurred;
	cv::Mat imgThresh;
	cv::Mat imgThreshCopy;
	cv::Mat imgThreshCopy2;
	cv::Mat imgThreshCopy3;
	cv::Mat imgThreshCopy4;
	cv::Mat imgThreshCopy5;


	std::vector<std::vector<cv::Point> > ptContours;
	std::vector<cv::Vec4i> v4iHierarchy;

	cv::Mat matClassificationInts;

	cv::Mat matTrainingImagesAsFlattenedFloats;

	std::vector<int> intValidChars = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',' };

	trainingImage = cv::imread("ss1.png"); //Nuskaitom nuotrauka
	//trainingImage.convertTo(trainingImage, -1, 1.3, -50);
	//cv::resize(trainingImage, trainingImage, cv::Size(), 0.55, 0.55);

	if (trainingImage.empty()) {
		std::cout << "error: image not read from file\n\n";
		return(0);
	}

	cv::cvtColor(trainingImage, imgGrayscale, CV_BGR2GRAY);

	cv::GaussianBlur(imgGrayscale,
		imgBlurred,
		cv::Size(5, 5),
		0);


	cv::adaptiveThreshold(imgBlurred,
		imgThresh,
		255,
		cv::ADAPTIVE_THRESH_GAUSSIAN_C,
		cv::THRESH_BINARY_INV,
		11,
		2);

	cv::imshow("imgThresh", imgThresh);
	imgThreshCopy = imgThresh.clone();
	imgThreshCopy2 = imgThresh.clone();


	cv::morphologyEx(imgThreshCopy,
		imgThreshCopy2,
		cv::MORPH_OPEN,
		cv::Mat(),
		cv::Point(-1, -1),
		1);

	cv::morphologyEx(imgThreshCopy2,
		imgThreshCopy3,
		cv::MORPH_CLOSE,
		cv::Mat(),
		cv::Point(-1, -1),
		1);

	cv::imshow("po morfologiniu", imgThreshCopy3);

	cv::findContours(imgThreshCopy3,
		ptContours,
		v4iHierarchy,
		cv::RETR_EXTERNAL,
		cv::CHAIN_APPROX_SIMPLE);


	for (int i = 0; i < ptContours.size(); i++) {
		if (cv::contourArea(ptContours[i]) > MIN_CONTOUR_AREA) {
			cv::Rect boundingRect = cv::boundingRect(ptContours[i]);

			cv::rectangle(trainingImage, boundingRect, cv::Scalar(0, 0, 255), 2);

			cv::Mat matROI = imgThreshCopy3(boundingRect);

			cv::Mat matROIResized;
			cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));

			cv::imshow("matROI", matROI);
			cv::imshow("matROIResized", matROIResized);
			cv::imshow("imgTrainingNumbers", trainingImage);

			int intChar = cv::waitKey(0);

			if (intChar == 27) {
				return(0);
			}
			else if (std::find(intValidChars.begin(), intValidChars.end(), intChar) != intValidChars.end()) {

				matClassificationInts.push_back(intChar);

				cv::Mat matImageFloat;
				matROIResized.convertTo(matImageFloat, CV_32FC1);

				cv::Mat matImageFlattenedFloat = matImageFloat.reshape(1, 1);

				matTrainingImagesAsFlattenedFloats.push_back(matImageFlattenedFloat);

			}
		}
	}

	std::cout << "training complete\n\n";


	cv::FileStorage fsClassifications("classifications2.xml", cv::FileStorage::WRITE);

	if (fsClassifications.isOpened() == false) {
		std::cout << "error, unable to open training classifications file, exiting program\n\n";
		return(0);
	}

	fsClassifications << "classifications" << matClassificationInts;
	fsClassifications.release();



	cv::FileStorage fsTrainingImages("images2.xml", cv::FileStorage::WRITE);

	if (fsTrainingImages.isOpened() == false) {
		std::cout << "error, unable to open training images file, exiting program\n\n";
		return(0);
	}

	fsTrainingImages << "images" << matTrainingImagesAsFlattenedFloats;
	fsTrainingImages.release();

	return(0);
}

