#include <stdlib.h>
#include <math.h>
#include <float.h>

double pair_corr_estimator(double u, double T, double h, int len_of_data, double data[])
{
    // uniform kernel estimator of pair correlation function
    // Illian et al. 2008, pp.230-232
    // Assume data is sorted in ascending order
    double estimator = 0;
    double diff = 0;

    for (int i = 0; i < len_of_data; i++)
    {
        for (int j = 0; j < len_of_data; j++)
        {
            if (i == j)
            {
                continue;
            }
            diff = data[j] - data[i]; // 単調増加
            if (diff - u >= h)
            {
                break;
            }
            if (diff - u >= -h)
            {
                estimator += 1 / (T - fabs(data[j] - data[i]));
            }
        }
    }
    estimator = estimator / (2 * h);
    estimator = estimator / pow(len_of_data / T, 2);
    return estimator;
}

double cross_corr_estimator(double u, double T, double h, int len_of_data1, int len_of_data2, double data1[], double data2[])
{
    // uniform kernel estimator of cross correlation function
    // Estimate correlation of (data2 - u) and data1
    // Assume data1 and data2 are sorted in ascending order
    double estimator = 0; // 最後に2hで割る
    double diff = 0;
    int start_j = 0;
    int flag = 0;
    for (int i = 0; i < len_of_data1; i++)
    {
        flag = 0; // Set flag = 1 when you find the first j that satisfies diff - r >= -h, and start the next search for j from start_j.
        for (int j = start_j; j < len_of_data2; j++)
        {
            diff = data2[j] - data1[i]; // 単調増加
            if (diff - u >= h)
            {
                break;
            }
            if (diff - u >= -h)
            {
                if (flag == 0)
                {
                    start_j = j;
                    flag = 1;
                }
                estimator += 1 / (T - fabs(data2[j] - data1[i]));
            }
        }
    }
    estimator = estimator / (2 * h);
    estimator = estimator / ((len_of_data1 / T) * (len_of_data2 / T));
    return estimator;
}
