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

    // Step X: QSAA Unit Test
    //   - Test QSAA at different loading times
    vector<int> load_time_s;
    for (int i = 10; i <= 1200; i = i + 10)
    {
        load_time_s.push_back(i);
    }
    vector<int> load_time_f;
    for (int i = 0; i < load_time_s.size(); i++)
    {
        int time_f = load_time_s.at(i) * 15;
        load_time_f.push_back(time_f);
        cout << time_f << endl;
    }
    vector<string> verification_res;
    vector<float> wpixels;
    string outputpath = parent_path + "Output/qsaa_load_time_f.txt";
    outfile.open(outputpath);
    outfile.close();

    for (int h = 0; h < load_time_f.size(); h++)
    {
        verification_res.clear();
        for (int i = 0; i < folder_path.size(); i++)
        {
            // Step X (Optional): Test Specific Angles
            //   1. Fetch QSAA vector data
            //   2. Run blink detection algorithm. Change Blink detection algorthm here.
            infile.open(parent_path + "QSAA Vector Data (Random)/" + folder_path[i].substr(69, 7) + "_" + "0x" + "," + "-45y" + ".txt");
            cout << "QSAA Vector Data (Random)/" + folder_path[i].substr(69, 7) + "_" + "0x" + "," + "-45y" + ".txt" << endl;
            wpixels.clear();
            string text = "";
            while (getline(infile, text)) {
                wpixels.push_back(stoi(text));
            }
            infile.close();
            string path = folder_path[i] + "0x" + "," + "-45y" + ".mp4";
            cout << path << endl;
            builder.BuildComputerVision(path, "0x", wpixels, load_time_f.at(h), "QSAA");

            // Step X: Fetch verification data from file
            infile.open(parent_path + "Blink In Validation Data/" + folder_path[i].substr(69, 7) + "_" + "0x" + "," + "-45y" + ".txt");
            blink_frameno.clear();
            text = "";
            while (getline(infile, text)) {
                blink_frameno.push_back(stoi(text));
            }
            infile.close();

            // Step X: Verify data from algorithm with data from file
            vector<int> predicted = builder._QSAA.blink_frameno;
            // _verify(outputpath, blink_frameno, builder._QSAA.blink_frameno);
            verification_res.push_back(_verify(blink_frameno, builder._QSAA.blink_frameno));
            // Step X (Optional): Testing All Angles
        }

        // Write Data to File
        outfile.open(outputpath, std::ios_base::app);
        outfile << load_time_f.at(h) << "\t";
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
