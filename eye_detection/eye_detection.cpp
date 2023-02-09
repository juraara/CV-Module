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
    vector<string> horizontal_path =
    {
        "E:/Online Class/1662543001899/Extras/1669357448407 - Eye Detection/Horizontal/Test-01/",
        "E:/Online Class/1662543001899/Extras/1669357448407 - Eye Detection/Horizontal/Test-02/",
        "E:/Online Class/1662543001899/Extras/1669357448407 - Eye Detection/Horizontal/Test-03/",
        "E:/Online Class/1662543001899/Extras/1669357448407 - Eye Detection/Horizontal/Test-04/",
        "E:/Online Class/1662543001899/Extras/1669357448407 - Eye Detection/Horizontal/Test-05/",
        "E:/Online Class/1662543001899/Extras/1669357448407 - Eye Detection/Horizontal/Test-06/"
    };
    vector<string> vertical_path =
    {
        "E:/Online Class/1662543001899/Extras/1669357448407 - Eye Detection/Vertical/Test-01/",
        "E:/Online Class/1662543001899/Extras/1669357448407 - Eye Detection/Vertical/Test-02/",
        "E:/Online Class/1662543001899/Extras/1669357448407 - Eye Detection/Vertical/Test-03/",
        "E:/Online Class/1662543001899/Extras/1669357448407 - Eye Detection/Vertical/Test-04/",
        "E:/Online Class/1662543001899/Extras/1669357448407 - Eye Detection/Vertical/Test-05/",
        "E:/Online Class/1662543001899/Extras/1669357448407 - Eye Detection/Vertical/Test-06/"
    };
    vector<string> distance =
    {
        "10cm",
        "20cm",
        "30cm"
    };
    vector<string> angle =
    {
        "-90",
        "-75",
        "-60",
        "-45",
        "-30",
        "-15",
        "0",
        "15",
        "30",
        "45",
        "60",
        "75",
        "90"
    };
    
    // Initialization
    ofstream outfile;
    ifstream infile;
    Builder builder;

    // Step X: Eye Detection (using Haar-Cascades)
    //   - Determine the limit of this eye detection algorithm at different angles
    vector<int> numofeyes = {1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1};
    vector<string> results;

    // Set output path
    string outputpath = parent_path + "Output/ed.txt";
    outfile.open(outputpath);
    outfile.close();
    
    cout << "Eye Detection (Horizontal Camera Angles)" << endl;
    for (int i = 0; i < distance.size(); i++)
    {
        outfile.open(outputpath, std::ios_base::app);
        outfile << "<Distance = " << distance.at(i) << ">" << endl;
        outfile.close();

        for (int j = 0; j < angle.size(); j++)
        {
            for (int k = 0; k < horizontal_path.size(); k++)
            {
                string path = horizontal_path[k] + distance[i] + "/" + angle[j] + ".png";
                cout << path << endl;

                // Call builder class
                builder.BuildComputerVision(path, "ED");

                // Verify
                int TP = 0, FP = 0, FN = 0;
                FP = builder._ED.numofeyes > numofeyes.at(j) ? builder._ED.numofeyes - numofeyes.at(j) : 0;
                TP = builder._ED.numofeyes > numofeyes.at(j) ? numofeyes.at(j) : builder._ED.numofeyes;
                FN = builder._ED.numofeyes > numofeyes.at(j) ? 0 : numofeyes.at(j) - builder._ED.numofeyes;
                char str[255];
                sprintf_s(str, "%d\t%d\t%d", TP, FP, FN);
                results.push_back(str);
            }
            
            // Write Data to File
            outfile.open(outputpath, std::ios_base::app);
            for (int k = 0; k < results.size(); k++)
            {
                outfile << results.at(k) << "\t";
            }
            results.clear();
            outfile << endl;
            outfile.close();
        }
    }

    /*string path = horizontal_path[0] + "30cm" + "/" + "_15" + ".png";
    cout << path << endl;
    // Call builder class
    builder.BuildComputerVision(path, "ED");*/

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
