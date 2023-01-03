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

#include "main.h"
#include "builder.h"

using namespace cv;
using namespace std;
using namespace ml;

int main()
{
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
    /*vector<int> minthresh;
    for (int i = 9; i <= 30; i = i + 1)
    {
        minthresh.push_back(i);
    }
    vector<string> verification_res;
    vector<float> wpixels;
    string outputpath = "Output/ceaa_minthresh.txt";
    outfile.open(outputpath);
    outfile.close();
    for (int k = 2; k < y_angle.size(); k++)
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
                infile.open("Blink In Validation Data/" + folder_path[i].substr(69, 7) + "_" + "0x" + "," + y_angle[k] + ".txt");
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
    }*/
    
    // Step X: QSAA Unit Test
    //   - Test QSAA at different loading times
    /*vector<int> load_time_s;
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
    string outputpath = "Output/qsaa_load_time_f.txt";
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
            infile.open("QSAA Vector Data (Random)/" + folder_path[i].substr(69, 7) + "_" + "0x" + "," + "-45y" + ".txt");
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
            infile.open("Blink In Validation Data/" + folder_path[i].substr(69, 7) + "_" + "0x" + "," + "-45y" + ".txt");
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
    }*/

    // Step X: GTON Eyelid Closure %Diff Test (Optimal Horizontal Angles Only)
    /*vector<double> eye_closure_perdiff;
    for (int i = 5; i <= 200; i = i + 10)
    {
        eye_closure_perdiff.push_back(i);
    }
    vector<string> verification_res;
    string outputpath = "Output/gton_eye_closure_perdiff.txt";
    outfile.open(outputpath);
    outfile.close();
    for (int k = 0; k < y_angle.size(); k++)
    {
        outfile.open(outputpath, std::ios_base::app);
        outfile << "<Angle (y) = " << y_angle[k] << ">" << endl;
        outfile.close();
        for (int h = 0; h < eye_closure_perdiff.size(); h++)
        {
            verification_res.clear();
            for (int i = 0; i < folder_path.size(); i++)
            {
                // Step X: Run blink detection algorithm. Change Blink detection algorthm here.
                string path = folder_path[i] + "0x" + "," + y_angle[k] + ".mp4";
                cout << path << endl;
                builder.BuildComputerVision(path, "0x", 80.0, -1, eye_closure_perdiff.at(h), "GTON");

                // Step X: Fetch verification data from file
                infile.open("Blink In Validation Data/" + folder_path[i].substr(69, 7) + "_" + "0x" + "," + y_angle[k] + ".txt");
                blink_frameno.clear();
                string text = "";
                while (getline(infile, text)) {
                    blink_frameno.push_back(stoi(text));
                }
                infile.close();

                // Step X: Verify data from algorithm with data from file
                verification_res.push_back(_verify(blink_frameno, builder._GTON.blink_frameno));

            }

            // Write Data to File
            outfile.open(outputpath, std::ios_base::app);
            outfile << eye_closure_perdiff.at(h) << "\t";
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
    }*/
    
    // Step X: GTON Eyelid Opening %Diff Test (Optimal Horizontal Angles Only)
    /*vector<double> eye_opening_perdiff;
    for (int i = -10; i <= -200; i = i - 10)
    {
        eye_opening_perdiff.push_back(i);
    }
    vector<string> verification_res;
    string outputpath = "Output/gton_eye_closure_perdiff.txt";
    outfile.open(outputpath);
    outfile.close();
    for (int k = 0; k < y_angle.size(); k++)
    {
        outfile.open(outputpath, std::ios_base::app);
        outfile << "<Angle (y) = " << y_angle[k] << ">" << endl;
        outfile.close();
        for (int h = 0; h < eye_opening_perdiff.size(); h++)
        {
            verification_res.clear();
            for (int i = 0; i < folder_path.size(); i++)
            {
                // Step X: Run blink detection algorithm. Change Blink detection algorthm here.
                string path = folder_path[i] + "0x" + "," + y_angle[k] + ".mp4";
                cout << path << endl;
                builder.BuildComputerVision(path, "0x", 80.0, -1, eye_opening_perdiff.at(h), "GTON");

                // Step X: Fetch verification data from file
                infile.open("Blink Out Validation Data/" + folder_path[i].substr(69, 7) + "_" + "0x" + "," + y_angle[k] + ".txt");
                blink_frameno.clear();
                string text = "";
                while (getline(infile, text)) {
                    blink_frameno.push_back(stoi(text));
                }
                infile.close();

                // Step X: Verify data from algorithm with data from file
                verification_res.push_back(_verify(blink_frameno, builder._GTON.blink_frameno));

            }

            // Write Data to File
            outfile.open(outputpath, std::ios_base::app);
            outfile << eye_closure_perdiff.at(h) << "\t";
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
    }*/

    // Step X: GTON Eyelid Closure and Opening %Diff Test
    /*vector<double> eye_opening_perdiff = {-10, -50, -100, -150, -200};
    vector<double> eye_closure_perdiff = {10, 50, 100, 150, 200};
    //   1. Eyelid Opening %Diff Test
    string outputpath = "Output/gton_eye_opening_perdiff.txt";
    outfile.open(outputpath);
    outfile.close();
    for (int h = 0; h < eye_opening_perdiff.size(); h++)
    {
        outfile.open(outputpath, std::ios_base::app);
        outfile << "% Diff = " << eye_opening_perdiff[h] << endl;
        outfile.close();
        for (int i = 0; i < folder_path.size(); i++)
        {
            outfile.open(outputpath, std::ios_base::app);
            outfile << "Test ID = " << folder_path[i].substr(69, 7) << endl;
            outfile << "TP\tFP\tFN" << endl;
            outfile.close();
            for (int j = 0; j < x_angle.size(); j++)
            {
                for (int k = 0; k < y_angle.size(); k++)
                {
                    // Step X: Run blink detection algorithm. Change Blink detection algorthm here.
                    string path = folder_path[i] + x_angle[j] + "," + y_angle[k] + ".mp4";
                    cout << path << endl;
                    builder.BuildComputerVision(path, x_angle[j], 80.0, eye_opening_perdiff.at(h), 1, "GTON");

                    // Step X: Fetch verification data from file
                    infile.open("Blink Out Validation Data/" + folder_path[i].substr(69, 7) + "_" + x_angle[j] + "," + y_angle[k] + ".txt");
                    blink_frameno.clear();
                    string text = "";
                    while (getline(infile, text)) {
                        blink_frameno.push_back(stoi(text));
                    }
                    infile.close();

                    // Step X: Verify data from algorithm with data from file
                    _verify(outputpath, blink_frameno, builder._GTON.blink_out_frameno);
                }
            }
        }
    }
    //   2. Eyelid Closure %Diff Test
    outputpath = "Output/gton_eye_closure_perdiff.txt";
    outfile.open(outputpath);
    outfile.close();
    for (int h = 0; h < eye_closure_perdiff.size(); h++)
    {
        outfile.open(outputpath, std::ios_base::app);
        outfile << "% Diff = " << eye_closure_perdiff[h] << endl;
        outfile.close();
        for (int i = 0; i < folder_path.size(); i++)
        {
            outfile.open(outputpath, std::ios_base::app);
            outfile << "Test ID = " << folder_path[i].substr(69, 7) << endl;
            outfile << "TP\tFP\tFN" << endl;
            outfile.close();
            for (int j = 0; j < x_angle.size(); j++)
            {
                for (int k = 0; k < y_angle.size(); k++)
                {
                    // Step X: Run blink detection algorithm. Change Blink detection algorthm here.
                    string path = folder_path[i] + x_angle[j] + "," + y_angle[k] + ".mp4";
                    cout << path << endl;
                    builder.BuildComputerVision(path, x_angle[j], 80.0, -1, eye_closure_perdiff.at(h), "GTON");

                    // Step X: Fetch verification data from file
                    infile.open("Blink In Validation Data/" + folder_path[i].substr(69, 7) + "_" + x_angle[j] + "," + y_angle[k] + ".txt");
                    blink_frameno.clear();
                    string text = "";
                    while (getline(infile, text)) {
                        blink_frameno.push_back(stoi(text));
                    }
                    infile.close();

                    // Step X: Verify data from algorithm with data from file
                    _verify(outputpath, blink_frameno, builder._GTON.blink_in_frameno);
                }
            }
        }
    }*/

    // Step X: Fill QSAA Vector Data (Random)
    /*RNG rng(12345);
    int load_time_f = 18000; // 20 minutes (15 fps)
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                string path = folder_path[i] + x_angle[j] + "," + y_angle[k] + ".mp4";
                cout << path << endl;
                builder.BuildComputerVision(path, x_angle[j], "ACC");
                outfile.open("QSAA Vector Data (Random)/" + folder_path[i].substr(69, 7) + "_" + x_angle[j] + "," + y_angle[k] + ".txt");
                for (int l = 0; l < load_time_f; l++)
                {
                    int random_num = rng.uniform(0, builder._ACC.wpixels.size() - 1);
                    outfile << builder._ACC.wpixels[random_num] << endl;
                }
                outfile.close();
            }
        }
    }*/

    // Step X: Fill QSAA Vector Data (Controlled)
    /*for (int i = 0; i < folder_path.size(); i++)
    {
        for (int j = 0; j < x_angle.size(); j++)
        {
            for (int k = 0; k < y_angle.size(); k++)
            {
                path = folder_path[i] + x_angle[j] + "," + y_angle[k] + ".mp4";
                cout << path << endl;
                builder.BuildComputerVision(path, x_angle[j], "ACC");
                outfile.open("QSAA Vector Data (Controlled)/" + folder_path[i].substr(69, 7) + "_" + x_angle[j] + "," + y_angle[k] + ".txt");
                
                // Step X: Sort vector in ascending order
                vector<float> sorted_wpixels = builder._ACC.wpixels;
                std::sort(sorted_wpixels.begin(), sorted_wpixels.end());

                // Step X: The average of the largest 5% of data is marked as O meaning open state.
                int vector_begin = builder._ACC.wpixels.size() - (builder._ACC.wpixels.size() / 10);
                int vector_end = builder._ACC.wpixels.size() - (builder._ACC.wpixels.size() / 20);
                float sum_of_elems = 0;
                for (int i = vector_begin; i <= vector_end; i++)
                {
                    sum_of_elems += sorted_wpixels[i];
                }
                int O = sum_of_elems / (builder._ACC.wpixels.size() / 20);

                // Step X: The average of the smallest 5% of data is marked as C meaning closed state.
                vector_begin = builder._ACC.wpixels.size() / 20;
                vector_end = builder._ACC.wpixels.size() / 10;
                sum_of_elems = 0;
                for (int i = vector_begin; i <= vector_end; i++)
                {
                    sum_of_elems += sorted_wpixels[i];
                }
                int C = sum_of_elems / (builder._ACC.wpixels.size() / 20);

                cout << O << "\t" << C << endl;

                for (int l = 0; l < 1800; l++)
                {
                    if (l < 900)
                    {
                        outfile << O << endl;
                    }
                    else
                    {
                        outfile << C << endl;
                    }
                }
                outfile.close();
            }
        }
    }*/

    // Step X: Global Thresholding of the Negative
    /*string outputpath = "Output/gton.txt";
    outfile.open(outputpath);
    outfile << "Algorithm ID = GTON" << endl;
    outfile.close();
    
    // for (int i = 0; i < folder_path.size(); i++)
    for (int i = 2; i < 3; i++)
    {
        outfile.open(outputpath, std::ios_base::app);
        outfile << "Test ID = " << folder_path[i].substr(69, 7) << endl;
        outfile << "TP\tFP\tFN" << endl;
        outfile.close();

        // for (int j = 1; j < 2; j++)
        for (int j = 0; j < x_angle.size(); j++)
        {
            // for (int k = 2; k < y_angle.size(); k++)
            for (int k = 0; k < y_angle.size(); k++)
            {
                // Step X: Run blink detection algorithm. Change Blink detection algorthm here.
                string path = folder_path[i] + x_angle[j] + "," + y_angle[k] + ".mp4";
                cout << path << endl;
                builder.BuildComputerVision(path, "0x", 80.0, -50, 50, "GTON");
                
                // Step X: Fetch verification data from file
                infile.open("Blink In Validation Data/" + folder_path[i].substr(69, 7) + "_" + x_angle[j] + "," + y_angle[k] + ".txt");
                blink_frameno.clear();
                string text = "";
                while (getline(infile, text)) {
                    blink_frameno.push_back(stoi(text));
                }
                infile.close();

                // Step X: Verify data from algorithm with data from file
                _verify(outputpath, blink_frameno, builder._GTON.blink_frameno);
            }
        }
    }*/

    // Step X: Quick Sort and Analysis
    /*vector<float> wpixels;
    outputpath = "Output/qsaa.txt";
    outfile.open(outputpath);
    outfile << "Algorithm ID = QSAA" << endl;
    outfile.close();
    for (int i = 0; i < folder_path.size(); i++)
    {
        outfile.open(outputpath, std::ios_base::app);
        outfile << "Test ID = " << folder_path[i].substr(69, 7) << endl;
        outfile << "TP\tFP\tFN" << endl;
        outfile.close();
        for (int j = 0; j < x_angle.size(); j++)
        {
            for (int k = 0; k < y_angle.size(); k++)
            {
                // Step X: 
                //   1. Fetch QSAA vector data
                //   2. Run blink detection algorithm. Change Blink detection algorthm here.
                infile.open("QSAA Vector Data (Random)/" + folder_path[i].substr(69, 7) + "_" + x_angle[j] + "," + y_angle[k] + ".txt");
                cout << "QSAA Vector Data (Random)/" + folder_path[i].substr(69, 7) + "_" + x_angle[j] + "," + y_angle[k] + ".txt" << endl;
                wpixels.clear();
                string text = "";
                while (getline(infile, text)) {
                    wpixels.push_back(stoi(text));
                }
                infile.close();
                string path = folder_path[i] + x_angle[j] + "," + y_angle[k] + ".mp4";
                cout << path << endl;
                builder.BuildComputerVision(path, x_angle[j], wpixels, 4500, "QSAA");

                // Step X: Fetch verification data from file
                infile.open("Blink In Validation Data/" + folder_path[i].substr(69, 7) + "_" + x_angle[j] + "," + y_angle[k] + ".txt");
                blink_frameno.clear();
                text = "";
                while (getline(infile, text)) {
                    blink_frameno.push_back(stoi(text));
                }
                infile.close();
                
                // Step X: Verify data from algorithm with data from file
                vector<int> predicted = builder._QSAA.blink_frameno;
                _verify(outputpath, blink_frameno, builder._QSAA.blink_frameno);
            }
        }
    }*/

    // Step X: Eye Black Pixel Ratio Analysis
    /*outputpath = "Output/ebra.txt";
    outfile.open(outputpath);
    outfile << "Algorithm ID = EBRA" << endl;
    outfile.close();
    for (int i = 0; i < 5; i++)
    {
        outfile.open(outputpath, std::ios_base::app);
        outfile << "Test ID = " << folder_path[i].substr(69, 7) << endl;
        outfile << "TP\tFP\tFN" << endl;
        outfile.close();
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                // Step X: Run blink detection algorithm. Change Blink detection algorthm here.
                string path = folder_path[i] + x_angle[j] + "," + y_angle[k] + ".mp4";
                cout << path << endl;
                builder.BuildComputerVision(path, x_angle[j], "EBRA");

                // Step X: Fetch verification data from file
                infile.open("Blink In Validation Data/" + folder_path[i].substr(69, 7) + "_" + x_angle[j] + "," + y_angle[k] + ".txt");
                blink_frameno.clear();
                string text = "";
                while (getline(infile, text)) {
                    blink_frameno.push_back(stoi(text));
                }
                infile.close();

                // Step X: Verify data from algorithm with data from file
                _verify(outputpath, blink_frameno, builder._EBRA.blink_frameno);
            }
        }
    }*/

    // Step X: Contour Extraction and Analysis
    /*outputpath = "Output/ceaa.txt";
    outfile.open(outputpath);
    outfile << "Algorithm ID = CEAA" << endl;
    outfile.close();
    for (int i = 0; i < 5; i++)
    {
        outfile.open(outputpath, std::ios_base::app);
        outfile << "Test ID = " << folder_path[i].substr(69, 7) << endl;
        outfile << "TP\tFP\tFN" << endl;
        outfile.close();
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                // Step X: Run blink detection algorithm. Change Blink detection algorthm here.
                string path = folder_path[i] + x_angle[j] + "," + y_angle[k] + ".mp4";
                // path = folder_path[0] + x_angle[1] + "," + y_angle[0] + ".mp4";
                cout << path << endl;
                builder.BuildComputerVision(path, x_angle[j], 6, "CEAA");

                // Step X: Fetch verification data from file
                infile.open("Blink In Validation Data/" + folder_path[i].substr(69, 7) + "_" + x_angle[j] + "," + y_angle[k] + ".txt");
                blink_frameno.clear();
                string text = "";
                while (getline(infile, text)) {
                    blink_frameno.push_back(stoi(text));
                }
                infile.close();

                // Step X: Verify data from algorithm with data from file
                _verify(outputpath, blink_frameno, builder._CEAA.blink_frameno);
            }
        }
    }*/
    
    // Random Num Gen
    /*cout << "Chrono Random Generator" << endl;
    for (int i = 0; i < 100; i++) {
        std::cout << random(1, 100) << std::endl;
    }
    cout << "RNG (OpenCV) Random Generator" << endl;
    RNG rng(12345);
    for (int i = 0; i < 100; i++) {
        cout << rng.uniform(0, 100) << endl;
    }*/

    string path = folder_path[0] + "0x" + "," + "-15y" + ".mp4";
    cout << path << endl;
    builder.BuildComputerVision(path, "0x", 80.0, -50, 50, "GTON");

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
