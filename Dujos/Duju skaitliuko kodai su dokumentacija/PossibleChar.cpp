// PossibleChar.cpp

#include "PossibleChar.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
PossibleChar::PossibleChar(std::vector<cv::Point> _contour) {
	///paduodamas konturas i metoda
    contour = _contour;
	///Priskiriamas apvadas musu turimam konturui
    boundingRect = cv::boundingRect(contour);
	///Centrine x koordinate
    intCenterX = (boundingRect.x + boundingRect.x + boundingRect.width) / 2;
	///Centrine y koordinate
    intCenterY = (boundingRect.y + boundingRect.y + boundingRect.height) / 2;

}

