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

struct  {
    vector<string> horizontal_path =
    {
        "G:/Online Class/1662543001899/CPE 4101L - CPE DESIGN 2/Program/Testing Data/TD 01 - ED Camera Angles/Horizontal/Test-01/",
        "G:/Online Class/1662543001899/CPE 4101L - CPE DESIGN 2/Program/Testing Data/TD 01 - ED Camera Angles/Horizontal/Test-02/",
        "G:/Online Class/1662543001899/CPE 4101L - CPE DESIGN 2/Program/Testing Data/TD 01 - ED Camera Angles/Horizontal/Test-03/",
        "G:/Online Class/1662543001899/CPE 4101L - CPE DESIGN 2/Program/Testing Data/TD 01 - ED Camera Angles/Horizontal/Test-04/",
        "G:/Online Class/1662543001899/CPE 4101L - CPE DESIGN 2/Program/Testing Data/TD 01 - ED Camera Angles/Horizontal/Test-05/",
        "G:/Online Class/1662543001899/CPE 4101L - CPE DESIGN 2/Program/Testing Data/TD 01 - ED Camera Angles/Horizontal/Test-06/"
    };
    vector<string> vertical_path =
    {
        "G:/Online Class/1662543001899/CPE 4101L - CPE DESIGN 2/Program/Testing Data/TD 01 - ED Camera Angles/Vertical/Test-01/",
        "G:/Online Class/1662543001899/CPE 4101L - CPE DESIGN 2/Program/Testing Data/TD 01 - ED Camera Angles/Vertical/Test-02/",
        "G:/Online Class/1662543001899/CPE 4101L - CPE DESIGN 2/Program/Testing Data/TD 01 - ED Camera Angles/Vertical/Test-03/",
        "G:/Online Class/1662543001899/CPE 4101L - CPE DESIGN 2/Program/Testing Data/TD 01 - ED Camera Angles/Vertical/Test-04/",
        "G:/Online Class/1662543001899/CPE 4101L - CPE DESIGN 2/Program/Testing Data/TD 01 - ED Camera Angles/Vertical/Test-05/",
        "G:/Online Class/1662543001899/CPE 4101L - CPE DESIGN 2/Program/Testing Data/TD 01 - ED Camera Angles/Vertical/Test-06/"
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
} EyeDetection;

struct {
    vector<string> folder_path =
    {
        "G:/Online Class/1662543001899/CPE 4101L - CPE DESIGN 2/Program/Testing Data/TD 02 - BD Camera Angles/Test-01/",
        "G:/Online Class/1662543001899/CPE 4101L - CPE DESIGN 2/Program/Testing Data/TD 02 - BD Camera Angles/Test-02/",
        "G:/Online Class/1662543001899/CPE 4101L - CPE DESIGN 2/Program/Testing Data/TD 02 - BD Camera Angles/Test-03/",
        "G:/Online Class/1662543001899/CPE 4101L - CPE DESIGN 2/Program/Testing Data/TD 02 - BD Camera Angles/Test-04/",
        "G:/Online Class/1662543001899/CPE 4101L - CPE DESIGN 2/Program/Testing Data/TD 02 - BD Camera Angles/Test-05/"
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
} BlinkDetection;

struct {
    vector<string> folder_path =
    {
        "E:/Online Class/1662543001899/Extras/1672213721086 - GTON (PERCLOS) (Unit Test)/Test-01/",
        "E:/Online Class/1662543001899/Extras/1672213721086 - GTON (PERCLOS) (Unit Test)/Test-02/",
        "E:/Online Class/1662543001899/Extras/1672213721086 - GTON (PERCLOS) (Unit Test)/Test-03/",
    };
} Perclos;

ofstream outfile;
ifstream infile;

void eyedetection_unittest(string output_fpath)
{
    // Step X: Eye Detection (using Haar-Cascades)
    //   - Determine the limit of this eye detection algorithm at different angles
    vector<int> numofeyes = { 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1 };
    vector<string> results;

    // Set output path
    outfile.open(output_fpath);
    outfile.close();

    cout << "Eye Detection (Horizontal Camera Angles)" << endl;
    for (int i = 0; i < EyeDetection.distance.size(); i++)
    {
        outfile.open(output_fpath, std::ios_base::app);
        outfile << "<Distance = " << EyeDetection.distance.at(i) << ">" << endl;
        outfile.close();

        for (int j = 0; j < EyeDetection.angle.size(); j++)
        {
            for (int k = 0; k < EyeDetection.horizontal_path.size(); k++)
            {
                string path = EyeDetection.horizontal_path[k] + EyeDetection.distance[i] + "/" + EyeDetection.angle[j] + ".png";
                cout << path << endl;

                // Call builder class
                Builder builder;
                builder.buildCV(path, "ED");

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
            outfile.open(output_fpath, std::ios_base::app);
            for (int k = 0; k < results.size(); k++)
            {
                outfile << results.at(k) << "\t";
            }
            results.clear();
            outfile << endl;
            outfile.close();
        }
    }
}

void qsaa_fillvectordat_random(string vectordat_fpath)
{
    RNG rng(12345);
    int load_time_f = 18000; // 20 minutes (15 fps)
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                string path = BlinkDetection.folder_path[i] + BlinkDetection.x_angle[j] + "," + BlinkDetection.y_angle[k] + ".mp4";
                cout << path << endl;
                Builder builder;
                builder.buildCV(path, BlinkDetection.x_angle[j], "ACC");
                outfile.open(vectordat_fpath + BlinkDetection.folder_path[i].substr(69, 7) + "_" + BlinkDetection.x_angle[j] + "," + BlinkDetection.y_angle[k] + ".txt");
                for (int l = 0; l < load_time_f; l++)
                {
                    int random_num = rng.uniform(0, builder._ACC.wpixels.size() - 1);
                    outfile << builder._ACC.wpixels[random_num] << endl;
                }
                outfile.close();
            }
        }
    }
}

void qsaa_fillvectordat_controlled(string vectordat_fpath)
{
    for (int i = 0; i < BlinkDetection.folder_path.size(); i++)
    {
        for (int j = 0; j < BlinkDetection.x_angle.size(); j++)
        {
            for (int k = 0; k < BlinkDetection.y_angle.size(); k++)
            {
                string path = BlinkDetection.folder_path[i] + BlinkDetection.x_angle[j] + "," + BlinkDetection.y_angle[k] + ".mp4";
                cout << path << endl;
                Builder builder;
                builder.buildCV(path, BlinkDetection.x_angle[j], "ACC");
                outfile.open(vectordat_fpath + BlinkDetection.folder_path[i].substr(69, 7) + "_" + BlinkDetection.x_angle[j] + "," + BlinkDetection.y_angle[k] + ".txt");

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
    }
}

// Step X: CEAA Unit Test (Optional)
//   - Determine the optimal minimum threshold for binary inversion
//   - The criteria for choosing the minimum threshold should is that there should
//     be no white pixels present during eyelid closure.
void ceaa_unittest(string blinkin_validationdat_fpath, string output_fpath)
{
    vector<int> minthresh;
    for (int i = 9; i <= 30; i = i + 1)
    {
        minthresh.push_back(i);
    }
    vector<string> misc_verification_res;
    outfile.open(output_fpath);
    outfile.close();
    for (int k = 2; k < BlinkDetection.y_angle.size(); k++)
    {
        outfile.open(output_fpath, std::ios_base::app);
        outfile << "<Angle (y) = " << BlinkDetection.y_angle[k] << ">" << endl;
        outfile.close();
        for (int h = 0; h < minthresh.size(); h++)
        {
            misc_verification_res.clear();
            for (int i = 0; i < BlinkDetection.folder_path.size(); i++)
            {
                // Step X: Run blink detection algorithm. Change Blink detection algorthm here.
                string path = BlinkDetection.folder_path[i] + "0x" + "," + BlinkDetection.y_angle[k] + ".mp4";
                // path = BlinkDetection.folder_path[0] + BlinkDetection.x_angle[1] + "," + BlinkDetection.y_angle[0] + ".mp4";
                cout << path << endl;
                Builder builder;
                builder.buildCV(path, "0x", minthresh[h], "CEAA");

                // Step X: Fetch verification data from file
                infile.open(blinkin_validationdat_fpath + BlinkDetection.folder_path[i].substr(69, 7) + "_" + "0x" + "," + BlinkDetection.y_angle[k] + ".txt");
                vector<int> blink_frameno;
                blink_frameno.clear();
                string text = "";
                while (getline(infile, text)) {
                    blink_frameno.push_back(stoi(text));
                }
                infile.close();

                // Step X: Verify data from algorithm with data from file
                misc_verification_res.push_back(_verify(blink_frameno, builder._CEAA.blink_frameno));
            }

            // Write Data to File
            outfile.open(output_fpath, std::ios_base::app);
            outfile << minthresh.at(h) << "\t";
            outfile.close();

            for (int i = 0; i < misc_verification_res.size(); i++)
            {
                outfile.open(output_fpath, std::ios_base::app);
                outfile << misc_verification_res.at(i) << "\t";
                outfile.close();
            }
            outfile.open(output_fpath, std::ios_base::app);
            outfile << endl;
            outfile.close();
        }
    }
}

// Step X: QSAA Unit Test
//   - Test QSAA at different loading times
void qsaa_unittest(string blinkin_validationdat_fpath, string output_fpath, string qsaa_vectordat_fpath)
{
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
    vector<string> misc_verification_res;
    vector<float> wpixels;
    outfile.open(output_fpath);
    outfile.close();

    for (int h = 0; h < load_time_f.size(); h++)
    {
        misc_verification_res.clear();
        for (int i = 0; i < BlinkDetection.folder_path.size(); i++)
        {
            // Step X (Optional): Test Specific Angles
            //   1. Fetch QSAA vector data
            //   2. Run blink detection algorithm. Change Blink detection algorthm here.
            infile.open(qsaa_vectordat_fpath + BlinkDetection.folder_path[i].substr(69, 7) + "_" + "0x" + "," + "-45y" + ".txt");
            cout << qsaa_vectordat_fpath + BlinkDetection.folder_path[i].substr(69, 7) + "_" + "0x" + "," + "-45y" + ".txt" << endl;
            wpixels.clear();
            string text = "";
            while (getline(infile, text)) {
                wpixels.push_back(stoi(text));
            }
            infile.close();
            string path = BlinkDetection.folder_path[i] + "0x" + "," + "-45y" + ".mp4";
            cout << path << endl;
            Builder builder;
            builder.buildCV(path, "0x", wpixels, load_time_f.at(h), "QSAA");

            // Step X: Fetch verification data from file
            infile.open(blinkin_validationdat_fpath + BlinkDetection.folder_path[i].substr(69, 7) + "_" + "0x" + "," + "-45y" + ".txt");
            vector<int> blink_frameno;
            blink_frameno.clear();
            text = "";
            while (getline(infile, text)) {
                blink_frameno.push_back(stoi(text));
            }
            infile.close();

            // Step X: Verify data from algorithm with data from file
            vector<int> predicted = builder._QSAA.blink_frameno;
            // _verify(output_fpath, blink_frameno, builder._QSAA.blink_frameno);
            misc_verification_res.push_back(_verify(blink_frameno, builder._QSAA.blink_frameno));
            // Step X (Optional): Testing All Angles
        }

        // Write Data to File
        outfile.open(output_fpath, std::ios_base::app);
        outfile << load_time_f.at(h) << "\t";
        outfile.close();

        for (int i = 0; i < misc_verification_res.size(); i++)
        {
            outfile.open(output_fpath, std::ios_base::app);
            outfile << misc_verification_res.at(i) << "\t";
            outfile.close();
        }
        outfile.open(output_fpath, std::ios_base::app);
        outfile << endl;
        outfile.close();
    }
}

// Step X: GTON Eyelid Closure %Diff Test (Optimal Horizontal Angles Only)
void gton_eye_closure_perdiff(string blinkin_validationdat_fpath, string output_fpath)
{
    vector<double> eye_closure_perdiff;
    for (int i = 5; i <= 200; i = i + 10)
    {
        eye_closure_perdiff.push_back(i);
    }
    vector<string> misc_verification_res;
    outfile.open(output_fpath);
    outfile.close();
    for (int k = 0; k < BlinkDetection.y_angle.size(); k++)
    {
        outfile.open(output_fpath, std::ios_base::app);
        outfile << "<Angle (y) = " << BlinkDetection.y_angle[k] << ">" << endl;
        outfile.close();
        for (int h = 0; h < eye_closure_perdiff.size(); h++)
        {
            misc_verification_res.clear();
            for (int i = 0; i < BlinkDetection.folder_path.size(); i++)
            {
                // Step X: Run blink detection algorithm. Change Blink detection algorthm here.
                string path = BlinkDetection.folder_path[i] + "0x" + "," + BlinkDetection.y_angle[k] + ".mp4";
                cout << path << endl;
                Builder builder;
                builder.buildCV(path, "0x", 80.0, -1, eye_closure_perdiff.at(h), "GTON");

                // Step X: Fetch verification data from file
                infile.open(blinkin_validationdat_fpath + BlinkDetection.folder_path[i].substr(69, 7) + "_" + "0x" + "," + BlinkDetection.y_angle[k] + ".txt");
                vector<int> blink_frameno;
                blink_frameno.clear();
                string text = "";
                while (getline(infile, text)) {
                    blink_frameno.push_back(stoi(text));
                }
                infile.close();

                // Step X: Verify data from algorithm with data from file
                misc_verification_res.push_back(_verify(blink_frameno, builder._GTON.blink_frameno));

            }

            // Write Data to File
            outfile.open(output_fpath, std::ios_base::app);
            outfile << eye_closure_perdiff.at(h) << "\t";
            outfile.close();

            for (int i = 0; i < misc_verification_res.size(); i++)
            {
                outfile.open(output_fpath, std::ios_base::app);
                outfile << misc_verification_res.at(i) << "\t";
                outfile.close();
            }
            outfile.open(output_fpath, std::ios_base::app);
            outfile << endl;
            outfile.close();
        }
    }
}

// Step X: GTON Eyelid Opening %Diff Test (Optimal Horizontal Angles Only)
void gton_eye_opening_perdiff(string blinkout_validationdat_fpath, string output_fpath)
{
    vector<double> eye_opening_perdiff;
    for (int i = -10; i <= -200; i = i - 10)
    {
        eye_opening_perdiff.push_back(i);
    }
    vector<string> misc_verification_res;
    outfile.open(output_fpath);
    outfile.close();
    for (int k = 0; k < BlinkDetection.y_angle.size(); k++)
    {
        outfile.open(output_fpath, std::ios_base::app);
        outfile << "<Angle (y) = " << BlinkDetection.y_angle[k] << ">" << endl;
        outfile.close();
        for (int h = 0; h < eye_opening_perdiff.size(); h++)
        {
            misc_verification_res.clear();
            for (int i = 0; i < BlinkDetection.folder_path.size(); i++)
            {
                // Step X: Run blink detection algorithm. Change Blink detection algorthm here.
                string path = BlinkDetection.folder_path[i] + "0x" + "," + BlinkDetection.y_angle[k] + ".mp4";
                cout << path << endl;
                Builder builder;
                builder.buildCV(path, "0x", 80.0, -1, eye_opening_perdiff.at(h), "GTON");

                // Step X: Fetch verification data from file
                infile.open(blinkout_validationdat_fpath + BlinkDetection.folder_path[i].substr(69, 7) + "_" + "0x" + "," + BlinkDetection.y_angle[k] + ".txt");
                vector<int> blink_frameno;
                blink_frameno.clear();
                string text = "";
                while (getline(infile, text)) {
                    blink_frameno.push_back(stoi(text));
                }
                infile.close();

                // Step X: Verify data from algorithm with data from file
                misc_verification_res.push_back(_verify(blink_frameno, builder._GTON.blink_frameno));

            }

            // Write Data to File
            outfile.open(output_fpath, std::ios_base::app);
            outfile << eye_opening_perdiff.at(h) << "\t";
            outfile.close();

            for (int i = 0; i < misc_verification_res.size(); i++)
            {
                outfile.open(output_fpath, std::ios_base::app);
                outfile << misc_verification_res.at(i) << "\t";
                outfile.close();
            }
            outfile.open(output_fpath, std::ios_base::app);
            outfile << endl;
            outfile.close();
        }
    }
}

void gton_perclos(string output_fpath)
{
    outfile.open(output_fpath);
    outfile.close();

    // Step X: Run blink detection algorithm. Change Blink detection algorthm here.
    string path = Perclos.folder_path[0] + "0x" + "," + "-30y" + ".mp4";
    cout << path << endl;
    Builder builder;
    builder.buildCV(path, "0x", 100.0, -50, 50, true, "GTON");

    // Write Data to File
    vector<int> blink_in_frameno = builder._GTON.blink_in_frameno;
    vector<int> blink_out_frameno = builder._GTON.blink_out_frameno;
    outfile.open(output_fpath, std::ios_base::app);
    outfile << "Test ID = Test-" << std::setfill('0') << std::setw(2) << 1 << endl;
    for (int j = 0; j < blink_in_frameno.size(); j++)
    {
        outfile << blink_in_frameno.at(j) << "\t" << blink_out_frameno.at(j) << endl;
    }
    outfile.close();
}

void gton_csi(string blinkin_validationdat_fpath, string output_fpath)
{
    outfile.open(output_fpath);
    outfile << "Algorithm ID = GTON" << endl;
    outfile.close();

    // for (int i = 0; i < BlinkDetection.folder_path.size(); i++)
    for (int i = 2; i < 3; i++)
    {
        outfile.open(output_fpath, std::ios_base::app);
        outfile << "Test ID = " << BlinkDetection.folder_path[i].substr(69, 7) << endl;
        outfile << "TP\tFP\tFN" << endl;
        outfile.close();

        // for (int j = 1; j < 2; j++)
        for (int j = 0; j < BlinkDetection.x_angle.size(); j++)
        {
            // for (int k = 2; k < BlinkDetection.y_angle.size(); k++)
            for (int k = 0; k < BlinkDetection.y_angle.size(); k++)
            {
                // Step X: Run blink detection algorithm. Change Blink detection algorthm here.
                string path = BlinkDetection.folder_path[i] + BlinkDetection.x_angle[j] + "," + BlinkDetection.y_angle[k] + ".mp4";
                cout << path << endl;
                Builder builder;
                builder.buildCV(path, "0x", 80.0, -50, 50, "GTON");

                // Step X: Fetch verification data from file
                infile.open(blinkin_validationdat_fpath + BlinkDetection.folder_path[i].substr(69, 7) + "_" + BlinkDetection.x_angle[j] + "," + BlinkDetection.y_angle[k] + ".txt");
                vector<int> blink_frameno;
                blink_frameno.clear();
                string text = "";
                while (getline(infile, text)) {
                    blink_frameno.push_back(stoi(text));
                }
                infile.close();

                // Step X: Verify data from algorithm with data from file
                _verify(output_fpath, blink_frameno, builder._GTON.blink_frameno);
            }
        }
    }
}

void qsaa_csi(string blinkin_validationdat_fpath, string output_fpath, string qsaa_vectordat_fpath)
{
    vector<float> wpixels;
    outfile.open(output_fpath);
    outfile << "Algorithm ID = QSAA" << endl;
    outfile.close();
    for (int i = 0; i < BlinkDetection.folder_path.size(); i++)
    {
        outfile.open(output_fpath, std::ios_base::app);
        outfile << "Test ID = " << BlinkDetection.folder_path[i].substr(69, 7) << endl;
        outfile << "TP\tFP\tFN" << endl;
        outfile.close();
        for (int j = 0; j < BlinkDetection.x_angle.size(); j++)
        {
            for (int k = 0; k < BlinkDetection.y_angle.size(); k++)
            {
                // Step X: 
                //   1. Fetch QSAA vector data
                //   2. Run blink detection algorithm. Change Blink detection algorthm here.
                infile.open(qsaa_vectordat_fpath + BlinkDetection.folder_path[i].substr(69, 7) + "_" + BlinkDetection.x_angle[j] + "," + BlinkDetection.y_angle[k] + ".txt");
                cout << qsaa_vectordat_fpath + BlinkDetection.folder_path[i].substr(69, 7) + "_" + BlinkDetection.x_angle[j] + "," + BlinkDetection.y_angle[k] + ".txt" << endl;
                wpixels.clear();
                string text = "";
                while (getline(infile, text)) {
                    wpixels.push_back(stoi(text));
                }
                infile.close();
                string path = BlinkDetection.folder_path[i] + BlinkDetection.x_angle[j] + "," + BlinkDetection.y_angle[k] + ".mp4";
                cout << path << endl;
                Builder builder;
                builder.buildCV(path, BlinkDetection.x_angle[j], wpixels, 4500, "QSAA");

                // Step X: Fetch verification data from file
                infile.open(blinkin_validationdat_fpath + BlinkDetection.folder_path[i].substr(69, 7) + "_" + BlinkDetection.x_angle[j] + "," + BlinkDetection.y_angle[k] + ".txt");
                vector<int> blink_frameno;
                blink_frameno.clear();
                text = "";
                while (getline(infile, text)) {
                    blink_frameno.push_back(stoi(text));
                }
                infile.close();

                // Step X: Verify data from algorithm with data from file
                vector<int> predicted = builder._QSAA.blink_frameno;
                _verify(output_fpath, blink_frameno, builder._QSAA.blink_frameno);
            }
        }
    }
}

void ebpra_csi(string blinkin_validationdat_fpath, string output_fpath)
{
    outfile.open(output_fpath);
    outfile << "Algorithm ID = EBRA" << endl;
    outfile.close();
    for (int i = 0; i < 5; i++)
    {
        outfile.open(output_fpath, std::ios_base::app);
        outfile << "Test ID = " << BlinkDetection.folder_path[i].substr(69, 7) << endl;
        outfile << "TP\tFP\tFN" << endl;
        outfile.close();
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                // Step X: Run blink detection algorithm. Change Blink detection algorthm here.
                string path = BlinkDetection.folder_path[i] + BlinkDetection.x_angle[j] + "," + BlinkDetection.y_angle[k] + ".mp4";
                cout << path << endl;
                Builder builder;
                builder.buildCV(path, BlinkDetection.x_angle[j], "EBRA");

                // Step X: Fetch verification data from file
                infile.open(blinkin_validationdat_fpath + BlinkDetection.folder_path[i].substr(69, 7) + "_" + BlinkDetection.x_angle[j] + "," + BlinkDetection.y_angle[k] + ".txt");
                vector<int> blink_frameno;
                blink_frameno.clear();
                string text = "";
                while (getline(infile, text)) {
                    blink_frameno.push_back(stoi(text));
                }
                infile.close();

                // Step X: Verify data from algorithm with data from file
                _verify(output_fpath, blink_frameno, builder._EBRA.blink_frameno);
            }
        }
    }
}

void ceaa_csi(string blinkin_validationdat_fpath, string output_fpath)
{
    outfile.open(output_fpath);
    outfile << "Algorithm ID = CEAA" << endl;
    outfile.close();
    for (int i = 0; i < 5; i++)
    {
        outfile.open(output_fpath, std::ios_base::app);
        outfile << "Test ID = " << BlinkDetection.folder_path[i].substr(69, 7) << endl;
        outfile << "TP\tFP\tFN" << endl;
        outfile.close();
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                // Step X: Run blink detection algorithm. Change Blink detection algorthm here.
                string path = BlinkDetection.folder_path[i] + BlinkDetection.x_angle[j] + "," + BlinkDetection.y_angle[k] + ".mp4";
                // path = BlinkDetection.folder_path[0] + BlinkDetection.x_angle[1] + "," + BlinkDetection.y_angle[0] + ".mp4";
                cout << path << endl;
                Builder builder;
                builder.buildCV(path, BlinkDetection.x_angle[j], 6, "CEAA");

                // Step X: Fetch verification data from file
                infile.open(blinkin_validationdat_fpath + BlinkDetection.folder_path[i].substr(69, 7) + "_" + BlinkDetection.x_angle[j] + "," + BlinkDetection.y_angle[k] + ".txt");
                vector<int> blink_frameno;
                blink_frameno.clear();
                string text = "";
                while (getline(infile, text)) {
                    blink_frameno.push_back(stoi(text));
                }
                infile.close();

                // Step X: Verify data from algorithm with data from file
                _verify(output_fpath, blink_frameno, builder._CEAA.blink_frameno);
            }
        }
    }
}