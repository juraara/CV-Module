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

#include "builder.h"

#define POS_TOLERANCE 5
#define NEG_TOLERANCE 5

using namespace cv;
using namespace std;
using namespace ml;

string _verify(vector<int> observed, vector<int> predicted)
{
    int TP = 0, FP = 0, FN = 0;
    int all = observed.size();
    // Step X: Count number of True Positives
    for (int i = 0; i < predicted.size(); i++)
    {
        int miss_count = 0;
        for (int j = 0; j < observed.size(); j++)
        {
            if (predicted[i] <= observed[j] + POS_TOLERANCE &&
                predicted[i] >= observed[j] - NEG_TOLERANCE)
            {
                cout << observed[j] << "\t";
                observed.erase(observed.begin() + j);
                j = 0;
                cout << predicted[i] << "\t" << observed.size() << endl;
                TP++;
                miss_count = 0;
                break;
            }
            else
            {
                miss_count++;
            }
        }
        if (miss_count >= observed.size())
        {
            if (miss_count != 0)
            {
                cout << "FP = " << predicted[i] << endl;
                FP++;
            }
        }
    }

    FN = all - TP;
    cout << TP << "\t" << FP << "\t" << FN << endl;
    char str[255];
    sprintf_s(str, "%d\t%d\t%d", TP, FP, FN);
    return str;
}

void _verify(string outputpath, vector<int> observed, vector<int> predicted)
{
    int TP = 0, FP = 0, FN = 0;
    int all = observed.size();
    // Step X: Count number of True Positives
    for (int i = 0; i < predicted.size(); i++)
    {
        int miss_count = 0;
        for (int j = 0; j < observed.size(); j++)
        {
            if (predicted[i] <= observed[j] + POS_TOLERANCE &&
                predicted[i] >= observed[j] - NEG_TOLERANCE)
            {
                cout << observed[j] << "\t";
                observed.erase(observed.begin() + j);
                j = 0;
                cout << predicted[i] << "\t" << observed.size() << endl;
                TP++;
                miss_count = 0;
                break;
            }
            else
            {
                miss_count++;
            }
        }
        if (miss_count >= observed.size())
        {
            if (miss_count != 0)
            {
                cout << "FP = " << predicted[i] << endl;
                FP++;
            }
        }
    }

    FN = all - TP;
    ofstream outfile;
    cout << TP << "\t" << FP << "\t" << FN << endl;
    outfile.open(outputpath, std::ios_base::app);
    outfile << TP << "\t" << FP << "\t" << FN << endl;
    outfile.close();
}

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine gen(seed);

int random(int low, int high)
{
    std::uniform_int_distribution<> dist(low, high);
    return dist(gen);
}