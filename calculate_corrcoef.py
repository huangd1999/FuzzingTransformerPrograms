
import numpy as np
Sort=[36,0,0,131,0,0,260,0,0,1037,0,0 ,]
Sort=[12,2,2,54,7,9,91,14,18,450,56,63,]

Reverse=[32,7,7,140,32,33,238,40,42,1032,76,80,]
Reverse=[13,2,2,46,7,8,86,10,13,427,55,65,]

Most=[34,18,20,158,86,91,274,160,177,1346,832,861,]
Most=[13,7,7,54,29,30,94,48,51,476,296,303,]

Dyck1=[37,4,4,184,17,22,367,26,41,1983,138,153,]
Dyck1=[10,6,7,50,21,22,98,37,38,556,210,213,]

Dyck2=[43,7,8,211,39,42,430,67,73,2691,521,529,]
Dyck2=[10,5,5,49,22,23,97,50,51,585,288,290,]

Histogram=[33,0,1,138,0,1,271,0,1,1346,1,2,]
Histogram=[13,0,0,54,0,0,99,0,0,489,1,1,]

Double=[50,32,32,223,118,119,411,214,215,2051,1073,1103,]
Double=[13,5,5,59,29,30,101,51,53,522,274,277,]

Conll=[54,32,33,247,148,150,492,292,299,2666,1624,1841,]
Conll=[11,7,9,52,27,31,102,65,73,568,341,478,]



Sort_p=[87.65 , 54.32 , 66.70 , 74.18 , 96.56]
Sort_p=[73.54, 50.08 , 61.04 , 72.03 , 83.07 ,]

Reverse_p=[90.25 , 58.22 , 69.38 , 78.41 , 98.37 ,]
Reverse_p=[80.17, 52.45 , 63.58 , 73.64 , 84.62 ,]

Most_p=[93.22 , 57.24 , 68.28 , 78.33 , 97.48 ,]
Most_p=[73.61, 51.52 , 62.54 , 72.58 , 83.53 ,]

Dyck1_p=[67.18 , 42.75 , 53.78 , 63.74 , 73.72 ,]
Dyck1_p=[67.18, 42.68 , 53.77 , 63.73 , 74.71 ,]

Dyck2_p=[76.17 , 50.67 , 61.65 , 71.68 , 90.63 ,]
Dyck2_p=[76.17, 50.59 , 61.64 , 71.67 , 90.61 ,]

Histogram_p=[90.99 , 56.93 , 67.09 , 77.45 , 96.52 ,]
[67.33, 41.71 , 52.74 , 62.76 , 72.78 ,] # error in paper Tab 4. We ignore add it in the second row which cause the result is same as the first row. We should change the 0.86 to 0.76

Double_p=[88.70 , 68.69 , 78.67 , 88.65 , 98.63 ,]
Double_p=[76.64, 53.56 , 64.53 , 74.52 , 85.54 ,]

Conll_p=[98.68 , 77.21 , 87.24 , 97.27 , 99.29 ,]
Conll_p=[94.00, 70.63 , 81.59 , 91.55 , 95.58 ,]

data = np.array([Sort,Reverse,Most,Dyck1,Dyck2,Histogram,Double,Conll])
p = np.array([Sort_p,Reverse_p,Most_p,Dyck1_p,Dyck2_p,Histogram_p,Double_p,Conll_p])

for i in range(8):
    temp_data = []
    for k in range(1,len(data[i]),3):
        temp_data.append(data[i][k])
    # print(temp_data)
    # print(p[i][1:])
    corr_matrix = np.corrcoef(temp_data,p[i][1:])
    correlation_coefficient = corr_matrix[0, 1]
    print(f"Pearson correlation coefficient between test coverage and number of faults: {correlation_coefficient}")