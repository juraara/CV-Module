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
        "E:/Online Class/1662543001899/Extras/1669438682706 - Blink Detection/Test-01/",
        "E:/Online Class/1662543001899/Extras/1669438682706 - Blink Detection/Test-02/",
        "E:/Online Class/1662543001899/Extras/1669438682706 - Blink Detection/Test-03/",
        "E:/Online Class/1662543001899/Extras/1669438682706 - Blink Detection/Test-04/",
        "E:/Online Class/1662543001899/Extras/1669438682706 - Blink Detection/Test-05/"
    };
    vector<string> x_angle = 
    {
        "-75x",
        "0x",
        "75x"
    };
    vector<string> y_angle = 
    {
        "-45y",
        "-15y",
        "15y"
    };
    
    // Initialization
    ofstream outfile;
    ifstream infile;
    Builder builder;
    vector<int> blink_frameno;

    // Step X: CEAA Unit Test (Optional)
    //   - Determine the optimal minimum threshold for binary inversion
    //   - The criteria for choosing the minimum threshold should is that there should
    //     be no white pixels present during eyelid closure.
    vector<int> minthresh;
    for (int i = 1; i <= 30; i = i + 1)
    {
        minthresh.push_back(i);
    }
    vector<string> verification_res;
    vector<float> wpixels;
    string outputpath = parent_path + "Output/ceaa_minthresh.txt";
    outfile.open(outputpath);
    outfile.close();
    for (int k = 1; k < y_angle.size(); k++)
    {
        outfile.open(outputpath, std::ios_base::app);
        outfile << "<Angle (y) = " << y_angle[k] << ">" << endl;
        outfile.close();
        for (int h = 0; h < minthresh.size(); h++)
        {
            verification_res.clear();
            for (int i = 0; i < folder_path.size(); i++)
            {
                // Step X: Run blink detection algorithm. Change Blink detection algorthm here.
                string path = folder_path[i] + "0x" + "," + y_angle[k] + ".mp4";
                // path = folder_path[0] + x_angle[1] + "," + y_angle[0] + ".mp4";
                cout << path << endl;
                builder.BuildComputerVision(path, "0x", minthresh[h], "CEAA");

                // Step X: Fetch verification data from file
                infile.open(parent_path + "Blink In Validation Data/" + folder_path[i].substr(69, 7) + "_" + "0x" + "," + y_angle[k] + ".txt");
                blink_frameno.clear();
                string text = "";
                while (getline(infile, text)) {
                    blink_frameno.push_back(stoi(text));
                }
                infile.close();

                // Step X: Verify data from algorithm with data from file
                verification_res.push_back(_verify(blink_frameno, builder._CEAA.blink_frameno));
            }

            // Write Data to File
            outfile.open(outputpath, std::ios_base::app);
            outfile << minthresh.at(h) << "\t";
            outfile.close();

            for (int i = 0; i < verification_res.size(); i++)
            {
                outfile.open(outputpath, std::ios_base::app);
                outfile << verification_res.at(i) << "\t";
                outfile.close();
            }
            outfile.open(outputpath, std::ios_base::app);
            outfile << endl;
            outfile.close();
        }
    }

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
