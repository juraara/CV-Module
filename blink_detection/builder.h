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

#include "gton.h"
#include "ceaa.h"
#include "ebpra.h"
#include "qsaa.h"
#include "acc.h"
#include "ed.h"

using namespace cv;
using namespace std;
using namespace ml;

class Builder
{
public:
    ACC _ACC;
    GTON _GTON;
    QSAA _QSAA;
    EBRA _EBRA;
    CEAA _CEAA;
    ED _ED;
    
    Builder() 
    {
    }

    void BuildComputerVision(string path, string id)
    {
        _ED = ED();
        
        CascadeClassifier eyeCascade;
        eyeCascade.load("haarcascade_eye_tree_eyeglasses.xml");
        CascadeClassifier faceCascade;
        faceCascade.load("haarcascade_frontalface_alt2.xml");

        // Configure here: cap(0) for webcam, cap((string)path) for videos
        VideoCapture cap(path);

        // Step X: Start Detecting Eyes
        Mat frame;
        Rect eye = Rect(0, 0, 0, 0);
        Rect iris = Rect(0, 0, 0, 0);

        if (id == "ED")
        {
            cap.read(frame);
            if (!frame.empty())
            {
                eye = _ED.detectEyes(frame, eyeCascade);
            }
        }
        imshow("frame", frame);
        waitKey(1);
        // Ask user to press any key to continue
        // system("pause");
    }

    // Contains GTON, EBRA, CEAA Default Configurations
    void BuildComputerVision(string path, string x_angle, string id)
    {
        _ACC = ACC();
        _GTON = GTON();
        _EBRA = EBRA();
        _CEAA = CEAA();

        CascadeClassifier eyeCascade;
        eyeCascade.load("haarcascade_eye_tree_eyeglasses.xml");
        CascadeClassifier faceCascade;
        faceCascade.load("haarcascade_frontalface_alt2.xml");

        // Configure here: cap(0) for webcam, cap((string)path) for videos
        VideoCapture cap(path);

        // Step X: Start Detecting Eyes
        Mat frame;
        Rect eye = Rect(0, 0, 0, 0);
        Rect iris = Rect(0, 0, 0, 0);
        while (eye.empty() || iris.empty())
        {
            // Change behaviour based on ID
            if (id == "ACC")
            {
                _ACC.frameno++;
                cap.read(frame);
                if (frame.empty()) break;
                eye = _ACC.detectEyes(frame, eyeCascade, x_angle);
                iris = _ACC.detectIris(frame, eye);
            }
            else if (id == "GTON")
            {
                _GTON.frameno++;
                cap.read(frame);
                if (frame.empty()) break;
                eye = _GTON.detectEyes(frame, eyeCascade, x_angle);
                iris = _GTON.detectIris(frame, eye);
            }
            else if (id == "EBRA")
            {
                _EBRA.frameno++;
                cap.read(frame);
                if (frame.empty()) break;
                eye = _EBRA.detectEyes(frame, eyeCascade, x_angle);
                iris = _EBRA.detectIris(frame, eye);
            }
            else if (id == "CEAA")
            {
                _CEAA.frameno++;
                cap.read(frame);
                if (frame.empty()) break;
                eye = _CEAA.detectEyes(frame, eyeCascade, x_angle);
                iris = _CEAA.detectIris(frame, eye);
            }

            imshow("detected", frame);
            waitKey(1);
        }
        // Step X: Start Detecting Blinks
        while (1)
        {
            // Change behaviour based on ID
            if (id == "ACC")
            {
                _ACC.frameno++;
                cap.read(frame);
                if (frame.empty()) break;
                frame = frame(eye);
                _ACC.detectBlink(frame, eye, iris);
            }
            else if (id == "GTON")
            {
                _GTON.frameno++;
                cap.read(frame);
                if (frame.empty()) break;
                frame = frame(eye);
                _GTON.detectBlink(frame, eye, iris);
            }
            else if (id == "EBRA")
            {
                _EBRA.frameno++;
                cap.read(frame);
                if (frame.empty()) break;
                frame = frame(eye);
                _EBRA.detectBlink(frame, eye, iris);
            }
            else if (id == "CEAA")
            {
                _CEAA.frameno++;
                cap.read(frame);
                if (frame.empty()) break;
                frame = frame(eye);
                _CEAA.detectBlink(frame, eye, iris);
            }

            waitKey(1);
        }
    }

    // QSAA Custom Config
    void BuildComputerVision(string path, string x_angle, vector<float> wpixels, int load_time_f, string id)
    {
        _QSAA = QSAA(wpixels, load_time_f);
        //_QSAA = QSAA();

        CascadeClassifier eyeCascade;
        eyeCascade.load("haarcascade_eye_tree_eyeglasses.xml");
        CascadeClassifier faceCascade;
        faceCascade.load("haarcascade_frontalface_alt2.xml");

        // Configure here: cap(0) for webcam, cap((string)path) for videos
        VideoCapture cap(path);

        // Start Detecting Eyes
        Mat frame;
        Rect eye = Rect(0, 0, 0, 0);
        Rect iris = Rect(0, 0, 0, 0);
        while (eye.empty() || iris.empty())
        {
            // Change behaviour based on ID
            if (id == "QSAA")
            {
                _QSAA.frameno++;
                cap.read(frame);
                if (frame.empty()) break;
                eye = _QSAA.detectEyes(frame, eyeCascade, x_angle);
                iris = _QSAA.detectIris(frame, eye);
            }

            imshow("detected", frame);
            waitKey(1);
        }
        // Start Detecting Blinks
        while (1)
        {
            // Change behaviour based on ID
            if (id == "QSAA")
            {
                _QSAA.frameno++;
                cap.read(frame);
                if (frame.empty()) break;
                frame = frame(eye);
                _QSAA.detectBlink(frame, eye, iris);
            }

            waitKey(1);
        }
    }
    
    // GTON Custom Config
    void BuildComputerVision(string path, string x_angle, double perclos_x, double eye_opening_perdiff, double eye_closure_perdiff, string id)
    {
        _GTON = GTON(perclos_x, eye_opening_perdiff, eye_closure_perdiff);

        CascadeClassifier eyeCascade;
        eyeCascade.load("haarcascade_eye_tree_eyeglasses.xml");
        CascadeClassifier faceCascade;
        faceCascade.load("haarcascade_frontalface_alt2.xml");

        // Configure here: cap(0) for webcam, cap((string)path) for videos
        VideoCapture cap(path);

        // Step X: Start Detecting Eyes
        Mat frame;
        Rect eye = Rect(0, 0, 0, 0);
        Rect iris = Rect(0, 0, 0, 0);
        while (eye.empty() || iris.empty())
        {
            // Change behaviour based on ID
            if (id == "GTON")
            {
                _GTON.frameno++;
                cap.read(frame);
                if (frame.empty()) break;
                eye = _GTON.detectEyes(frame, eyeCascade, x_angle);
                iris = _GTON.detectIris(frame, eye);
            }

            imshow("detected", frame);
            waitKey(1);
        }
        // Step X: Start Detecting Blinks
        while (1)
        {
            // Change behaviour based on ID
            if (id == "GTON")
            {
                _GTON.frameno++;
                cap.read(frame);
                if (frame.empty()) break;
                frame = frame(eye);
                _GTON.detectBlink(frame, eye, iris);
            }

            waitKey(1);
            system("pause");
        }
    }

    // GTON Custom Config (Rotate)
    //   - Config used for default camera placement (where the camera is positioned upside down) on the helmet
    void BuildComputerVision(string path, string x_angle, double perclos_x, double eye_opening_perdiff, double eye_closure_perdiff, bool rotate, string id)
    {
        _GTON = GTON(perclos_x, eye_opening_perdiff, eye_closure_perdiff);

        CascadeClassifier eyeCascade;
        eyeCascade.load("haarcascade_eye_tree_eyeglasses.xml");
        CascadeClassifier faceCascade;
        faceCascade.load("haarcascade_frontalface_alt2.xml");

        // Configure here: cap(0) for webcam, cap((string)path) for videos
        VideoCapture cap(path);

        // Step X: Start Detecting Eyes
        Mat frame;
        Rect eye = Rect(0, 0, 0, 0);
        Rect iris = Rect(0, 0, 0, 0);
        while (eye.empty() || iris.empty())
        {
            // Change behaviour based on ID
            if (id == "GTON")
            {
                _GTON.frameno++;
                cap.read(frame);
                if (frame.empty()) break;
                if (rotate)
                {
                    frame = _GTON.rotate(frame, 180);
                }
                eye = _GTON.detectEyes(frame, eyeCascade, x_angle);
                iris = _GTON.detectIris(frame, eye);
            }

            imshow("detected", frame);
            waitKey(1);
        }
        // Step X: Start Detecting Blinks
        while (1)
        {
            // Change behaviour based on ID
            if (id == "GTON")
            {
                _GTON.frameno++;
                cap.read(frame);
                if (frame.empty()) break;
                if (rotate)
                {
                    frame = _GTON.rotate(frame, 180);
                }
                frame = frame(eye);
                _GTON.detectBlink(frame, eye, iris);
            }

            waitKey(1);
        }
    }

    // CEAA Custom Config
    void BuildComputerVision(string path, string x_angle, double minthresh, string id)
    {
        _CEAA = CEAA(minthresh);

        CascadeClassifier eyeCascade;
        eyeCascade.load("haarcascade_eye_tree_eyeglasses.xml");
        CascadeClassifier faceCascade;
        faceCascade.load("haarcascade_frontalface_alt2.xml");

        // Configure here: cap(0) for webcam, cap((string)path) for videos
        VideoCapture cap(path);

        // Step X: Start Detecting Eyes
        Mat frame;
        Rect eye = Rect(0, 0, 0, 0);
        Rect iris = Rect(0, 0, 0, 0);
        while (eye.empty() || iris.empty())
        {
            // Change behaviour based on ID
            if (id == "CEAA")
            {
                _CEAA.frameno++;
                cap.read(frame);
                if (frame.empty()) break;
                eye = _CEAA.detectEyes(frame, eyeCascade, x_angle);
                iris = _CEAA.detectIris(frame, eye);
            }

            imshow("detected", frame);
            waitKey(1);
        }
        // Step X: Start Detecting Blinks
        while (1)
        {
            // Change behaviour based on ID
            if (id == "CEAA")
            {
                _CEAA.frameno++;
                cap.read(frame);
                if (frame.empty()) break;
                frame = frame(eye);
                _CEAA.detectBlink(frame, eye, iris);
            }

            waitKey(1);
        }
    }
private:

};
