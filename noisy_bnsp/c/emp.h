#ifndef INCLUDED_emp_h_
#define INCLUDED_emp_h_

double pair_corr_estimator(double u, double T, double h, int len_of_data, double data[]);
double cross_corr_estimator(double u, double T, double h, int len_of_data1, int len_of_data2, double data1[], double data2[]);
double cross_stoyan_f(double s, double u, double T, int len_of_data1, int len_of_data2, double data1[], double data2[]);

#endif
