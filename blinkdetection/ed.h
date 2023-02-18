#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <fstream>

#include <numeric>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace ml;

class ED
{
public:
    int numofeyes;

    ED()
    {
        // public
        numofeyes = 0;
    }

    Rect getLeftmostEye(vector<Rect>& eyes)
    {
        int leftmost = 99999999;
        int leftmostIndex = -1;
        for (int i = 0; i < eyes.size(); i++) {
            if (eyes[i].tl().x < leftmost) {
                leftmost = eyes[i].tl().x;
                leftmostIndex = i;
            }
        }
        return eyes[leftmostIndex];
    }

    Rect detectEyes(Mat& frame, CascadeClassifier& eyeCascade)
    {
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY); // convert image to grayscale
        equalizeHist(gray, gray); // enchance image contrast
        // Detect Both Eyes
        vector<Rect> eyes;
        eyeCascade.detectMultiScale(gray, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(90, 90)); // eye size (Size(90,90)) is determined emperically based on eye distance
        ofstream outfile;

        // Store total number of eyes detected
        numofeyes = eyes.size();
        if (eyes.size() == 0)
        {
            cout << "# of Eyes == 0" << endl;
            return Rect(0, 0, 0, 0); // return empty rectangle
        }

        // Draw rectangle around the eyes
        for (Rect& eye : eyes) {
            rectangle(frame, eye.tl(), eye.br(), Scalar(0, 255, 0), 2); // draw rectangle around both eyes
        }
        return getLeftmostEye(eyes);
    }

private:

};