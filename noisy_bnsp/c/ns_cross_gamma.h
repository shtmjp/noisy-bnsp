// ifndef などの意味は以下を参照
// https://www.kinjo-u.ac.jp/ushida/tekkotsu/cpp/ifndef.html
#ifndef INCLUDED_ns_cross_gamma_h_
#define INCLUDED_ns_cross_gamma_h_

double bigamma_pdf(double x, double a1, double a2, double l1, double l2);
double cross_wl_gamma(double r, double T, double a, double a1, double a2, double l1, double l2, int len_of_data1, int len_of_data2, double data1[], double data2[]);
double ccf_gamma(double u, double a, double a1, double a2, double l1, double l2);
double integrated_ccf_gamma(double r, double a, double a1, double a2, double l1, double l2);

#endif
