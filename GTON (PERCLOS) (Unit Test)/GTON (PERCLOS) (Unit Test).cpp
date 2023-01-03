// This file contains the 'main' function. Program execution begins and ends there.

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
#include <random>
#include <chrono>

#include <numeric>
#include <vector>
#include <algorithm>

#include "C:/Users/Administrator/source/repos/Computer Vision Module - Smart Helmet/CV (Parent)/main.h"
#include "C:/Users/Administrator/source/repos/Computer Vision Module - Smart Helmet/CV (Parent)/builder.h"

using namespace cv;
using namespace std;
using namespace ml;

int main()
{
    string parent_path = "C:/Users/Administrator/source/repos/Computer Vision Module - Smart Helmet/CV (Parent)/";
    vector<string> folder_path =
    {
        "E:/Online Class/1662543001899/Extras/1672213721086 - GTON (PERCLOS) (Unit Test)/Test-01/",
        "E:/Online Class/1662543001899/Extras/1672213721086 - GTON (PERCLOS) (Unit Test)/Test-02/",
        "E:/Online Class/1662543001899/Extras/1672213721086 - GTON (PERCLOS) (Unit Test)/Test-03/",
    };
    
    // Initialization
    ofstream outfile;
    ifstream infile;
    Builder builder;
    vector<int> blink_frameno;

    // Step X: Labeling the blink detection algorithm's PERCLOS as 'predicted' and observered 
    // PERCLOS values as 'observed', show the:
    //   - Pearson Correlation Coefficient
    //   - RMSE value
    string outputpath = parent_path + "Output/gton_perclos.txt";
    outfile.open(outputpath);
    outfile.close();

    // Step X: Run blink detection algorithm. Change Blink detection algorthm here.
    string path = folder_path[0] + "0x" + "," + "-30y" + ".mp4";
    cout << path << endl;
    builder.BuildComputerVision(path, "0x", 100.0, -50, 50, true, "GTON");

    // Write Data to File
    vector<int> blink_in_frameno = builder._GTON.blink_in_frameno;
    vector<int> blink_out_frameno = builder._GTON.blink_out_frameno;
    outfile.open(outputpath, std::ios_base::app);
    outfile << "Test ID = Test-" << std::setfill('0') << std::setw(2) << 1 << endl;
    for (int j = 0; j < blink_in_frameno.size(); j++)
    {
        outfile << blink_in_frameno.at(j) << "\t" << blink_out_frameno.at(j) << endl;
    }
    outfile.close();

    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
