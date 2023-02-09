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

#define PERCLOS_LOADTIME_F 900
#define CROP_PERCENT 0.2
#define THRESHOLD 15
#define MIN_NUMOF_WPIXEL 0
#define MIN_IRISCONTOURAREA 0

using namespace cv;
using namespace std;
using namespace ml;

class GTON
{
public:
    int frameno;
    vector<int> blink_frameno;
    vector<int> blink_in_frameno;
    vector<int> blink_out_frameno;
    float PERCLOS;

    GTON()
    {
        // public
        frameno = -1;
        blink_frameno.clear();
        blink_in_frameno.clear();
        blink_out_frameno.clear();
        PERCLOS = 0.0;

        // private
        prev_numof_bwpixels = 0.0;
        curr_numof_bwpixels = 0.0;
        perclos_x = 80.0;
        eye_opening_perdiff = -60.0;
        eye_closure_perdiff = 80.0;
        eyestate = 0;
        prev_eyestate = 0;
        eyestates.clear();
    }

    GTON(double _perclos_x, double _eye_opening_perdiff, double _eye_closure_perdiff)
    {
        // public
        frameno = -1;
        blink_frameno.clear();
        blink_in_frameno.clear();
        blink_out_frameno.clear();
        PERCLOS = 0.0;

        // private
        prev_numof_bwpixels = 0.0;
        curr_numof_bwpixels = 0.0;
        perclos_x = _perclos_x;
        eye_opening_perdiff = _eye_opening_perdiff;
        eye_closure_perdiff = _eye_closure_perdiff;
        eyestate = 0;
        prev_eyestate = 0;
        eyestates.clear();
    }

    Mat rotate(Mat src, double angle) //rotate function returning mat object with parametres imagefile and angle    
    {
        Mat dst; //Mat object for output image file
        Point2f pt(src.cols / 2., src.rows / 2.); //point from where to rotate    
        Mat r = getRotationMatrix2D(pt, angle, 1.0); //Mat object for storing after rotation
        warpAffine(src, dst, r, Size(src.cols, src.rows)); ///applie an affine transforation to image.
        return dst; //returning Mat object for output image file
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

    Rect detectEyes(Mat& frame, CascadeClassifier& eyeCascade, string x_angle)
    {
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY); // convert image to grayscale
        equalizeHist(gray, gray); // enchance image contrast
        // Detect Both Eyes
        vector<Rect> eyes;
        eyeCascade.detectMultiScale(gray, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(90, 90)); // eye size (Size(90,90)) is determined emperically based on eye distance

        if (x_angle == "-75x" || x_angle == "75x")
        {
            if (eyes.size() == 0)
            {
                cout << "# of Eyes == 0" << endl;
                return Rect(0, 0, 0, 0); // return empty rectangle
            }
            else if (eyes.size() > 1)
            {
                cout << "# of Eyes > 1" << endl;
                return Rect(0, 0, 0, 0); // return empty rectangle
            }
        }
        else
        {
            if (eyes.size() != 2) { // if both eyes not detected
                cout << "# of Eyes != 2" << endl;
                return Rect(0, 0, 0, 0); // return empty rectangle
            }
        }

        for (Rect& eye : eyes) {
            rectangle(frame, eye.tl(), eye.br(), Scalar(0, 255, 0), 2); // draw rectangle around both eyes
        }

        imshow("frame", frame);
        return getLeftmostEye(eyes);
    }

    // Iris Detection Steps:
    //   1. Use detected eye, crop unwanted areas (i.e. eyebrows) by cropping the sides by x%
    //   2. Determine largest contour which is the pupil
    //   3. Limitation: Iris color should be on the black side 
    Rect detectIris(Mat& frame, Rect& eye)
    {
        if (eye.empty()) return Rect(0, 0, 0, 0);
        frame = frame(eye);
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        // Find contours
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY); // convert image to grayscale
        equalizeHist(gray, gray); // enchance image contrast
        Mat blur;
        GaussianBlur(gray, blur, Size(9, 9), 0); // blur image
        Mat thresh;
        threshold(blur, thresh, THRESHOLD, 255, THRESH_BINARY_INV); // convert to binary image

        // Crop Sides to Remove Eyebrows etc.
        int x = thresh.cols * CROP_PERCENT;
        int y = thresh.rows * CROP_PERCENT;
        int src_w = thresh.cols * (1 - (CROP_PERCENT * 2));
        int src_h = thresh.rows * (1 - (CROP_PERCENT * 2));
        Mat crop = thresh(Rect(x, y, src_w, src_h)); // crop side to remove eyebrows etc.

        Mat mask = cv::Mat::zeros(thresh.size(), CV_8U);
        mask(Rect(x, y, src_w, src_h)) = 255;
        imshow("mask", mask);

        Mat dstImage = cv::Mat::zeros(thresh.size(), CV_8U);
        thresh.copyTo(dstImage, mask);
        imshow("dstImage", dstImage);


        findContours(dstImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

        if (contours.size() == 0) {
            cout << "Eyeball not detected" << endl;
            return Rect(0, 0, 0, 0);
        }

        int maxarea_contour_id = getMaxAreaContourId(contours);
        vector<Point> it = contours[maxarea_contour_id];
        contours.clear();
        contours.push_back(it);

        if (contourArea(contours.at(0)) < MIN_IRISCONTOURAREA)
        {
            cout << "Eyeball not detected" << endl;
            return Rect(0, 0, 0, 0);
        }

        // Approximate contours to polygons + get bounding rects and circles
        vector<vector<Point> > contours_poly(contours.size());
        vector<Rect> boundRect(contours.size());
        vector<Point2f>center(contours.size());
        vector<float>radius(contours.size());

        for (int i = 0; i < contours.size(); i++)
        {
            approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
            boundRect[i] = boundingRect(Mat(contours_poly[i]));
            minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
        }

        // Draw polygonal contour + bonding rects + circles
        for (int i = 0; i < contours.size(); i++)
        {
            drawContours(frame, contours_poly, i, Scalar(95, 191, 0), 2, 8, vector<Vec4i>(), 0, Point());
            rectangle(frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 191), 2, 8, 0);
            // circle(frame, center[i], (int)radius[i], color, 2, 8, 0);
        }

        // Show in a window
        imshow("thresh", thresh);
        return boundRect[0];
    }

    void detectBlink(Mat& frame, Rect& eye, Rect& iris)
    {
        // BGR to Binary
        Mat gray;
        imshow("frame", frame);
        cvtColor(frame, gray, COLOR_BGR2GRAY); // convert image to grayscale
        equalizeHist(gray, gray); // enchance image contrast
        Mat blur;
        GaussianBlur(gray, blur, Size(9, 9), 0); // blur image
        Mat thresh;
        threshold(blur, thresh, THRESHOLD, 255, THRESH_BINARY_INV); // convert to binary image

        // Crop Sides to Remove Eyebrows etc.
        int x = thresh.cols * CROP_PERCENT;
        int y = thresh.rows * CROP_PERCENT;
        int src_w = thresh.cols * (1 - (CROP_PERCENT * 2));
        int src_h = thresh.rows * (1 - (CROP_PERCENT * 2));
        Mat crop = thresh(Rect(x, y, src_w, src_h)); // crop side to remove eyebrows etc.

        // Get Iris
        int iris_h = iris.br().y - iris.tl().y;

        // Get Upper Half of Cropped Frame
        int upper_w = crop.cols;
        // Double values: 
        //   1. 0.2 - 20%, p80 where <=20% of eye opening will be considered CLOSED STATE
        //   2. 0.5 - 50%, p50 where <=50% of eye opening will be considered CLOSED STATE
        // Original
        // int upper_h = iris.br().y - (thresh.rows * CROP_PERCENT) - (iris_h * (1.0 - PERCLOS_X / 100.0));
        // Mat upper = crop(Rect(0, 0, upper_w, upper_h)); // get upper half of image
        // Modified
        int upper_h = iris_h - (iris_h * (1.0 - perclos_x / 100.0));
        Mat upper = crop(Rect(0, (iris.tl().y - y), upper_w, upper_h)); // get upper half of image

        // <Image Pre-processing Ends Here>
        //   1. This part is where other blink detection methods starts to differ in algorithm

        // Calculate Histogram
        int histSize = 256;
        float range[] = { 0, 256 }; // the upper boundary is exclusive
        const float* histRange[] = { range };
        bool uniform = true, accumulate = false;
        Mat hist;
        calcHist(&upper, 1, 0, Mat(), hist, 1, &histSize, histRange, uniform, accumulate); // get histogram

        // Compare Current and Previous Frames
        prev_numof_bwpixels = curr_numof_bwpixels;
        curr_numof_bwpixels = hist.at<float>(255) < MIN_NUMOF_WPIXEL ? 0 : hist.at<float>(255);
        float percentDiff = ((prev_numof_bwpixels - curr_numof_bwpixels) / ((prev_numof_bwpixels + curr_numof_bwpixels) / 2)) * 100;
        // float percentChange = ((curr_numof_bwpixels - prev_numof_bwpixels) / prev_numof_bwpixels) * 100;
        cout << "Prev No. of White Pixels = " << prev_numof_bwpixels << endl;
        cout << "Curr No. of White Pixels = " << curr_numof_bwpixels << endl;
        cout << "Percent Diff = " << percentDiff << endl;

        prev_eyestate = eyestate;
        if (percentDiff >= eye_closure_perdiff) {
            eyestate = 1;
        }
        else if (percentDiff <= eye_opening_perdiff) {
            eyestate = 0;
        }

        // Record Blinks
        if (eyestate == 1)
        {
            if (eyestate != prev_eyestate)
            {
                blink_frameno.push_back(frameno);
                blink_in_frameno.push_back(frameno);
                // cout << frameno << "\t Percent Diff = " << percentDiff << endl;
            }
        }
        else if(eyestate == 0)
        {
            if (eyestate != prev_eyestate)
            {
                blink_out_frameno.push_back(frameno);
                // cout << "<Blink Out> " << frameno << endl;
            }
        }

        // Calculate PERCLOS
        calcPERCLOS(eyestate);

        // Draw Lines
        Point p1(x, y);
        Point p2(x + src_w, y + src_h);
        rectangle(frame, p1, p2, Scalar(0, 255, 0), 2);
        imshow("frame (marked) (r)", frame);
        rectangle(frame, Point(x, iris.tl().y), Point(x + src_w, iris.br().y), Scalar(0, 0, 255), 2);
        imshow("frame (marked) (rg)", frame);
        // line(frame, Point(x, iris.tl().y), Point(x + src_w, iris.tl().y), Scalar(0, 0, 255), 2, 8, 0);
        // line(frame, Point(x, iris.br().y), Point(x + src_w, iris.br().y), Scalar(0, 0, 255), 2, 8, 0);

        imshow("crop", crop);
        imshow("upper", upper);
        waitKey(1);
    }

    int getMaxAreaContourId(vector <vector<cv::Point>> contours)
    {
        double maxArea = 0;
        int maxAreaContourId = -1;
        for (int j = 0; j < contours.size(); j++)
        {
            double newArea = cv::contourArea(contours.at(j));
            if (newArea > maxArea)
            {
                maxArea = newArea;
                maxAreaContourId = j;
            } // End if
        } // End for
        return maxAreaContourId;
    }

    void calcPERCLOS(int& eyestate)
    {
        if (eyestates.size() == PERCLOS_LOADTIME_F)
        {
            // Step X: 
            //   - Get total number of frames in 'closed state' within
            //     the 1 minute window 
            //       - 1 minute window = 900 frames
            //       - closed state is when: eyestate = 1
            //   - Window is fixed in size and moves 1 frame at a time
            int closedstates = 0;
            closedstates = accumulate(eyestates.begin(), eyestates.end(), 0);
            PERCLOS = (float)closedstates / PERCLOS_LOADTIME_F * 100;

            // Step X: Shift left to remove least recent element
            std::rotate(eyestates.begin(), eyestates.begin() + 1, eyestates.end());
            eyestates.pop_back();
        }
        // Step X: Fill/refill vector 
        if (eyestate == 1)
        {
            eyestates.push_back(1);
        }
        else if (eyestate == 0)
        {
            eyestates.push_back(0);
        }
    }

private:
    float prev_numof_bwpixels;
    float curr_numof_bwpixels;
    double perclos_x;
    double eye_opening_perdiff;
    double eye_closure_perdiff;
    int eyestate;
    int prev_eyestate;
    vector<int> eyestates;
};
