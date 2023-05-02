# Preprocessing Variant `Some tries and first approach`

## Description
Started with classical pipeline for grid detection

## Pipeline
- [x] Gray level
- [x] Blur 3x3
- [x] Canny ***1**
- [x] Hough (rho and theta version)
- [x] Cluster lines ****2***
- [x] Lines intersection
- [ ] find 4 corners
- [ ] geometric tran1sformations
- [ ] extract 64(8x8) sub-img  

## Comments and notes
> ****1***
>> Upper threshold of Canny is obtained considering the median of grey level
>> Lower threshold is 1/3 of upper threshold as default recommended from openCV
>> Tried with simmetric thresholds of sigmas, no visible improvements
>> Tried different values of sigmas:
>> ```
>> for delta_sigma in range(10):    
>>    sigma = 0.09+0.03*delta_sigma
>> ```

> ****2***
>> Tried different approaches
>> Classical way: cluster points after intersecting the lines
>> Using: cluster lines with Kmeans (if more than 9 group of lines are found, troubles with kmeans) and then intersect point
>> TODO: try using agglomerative clustering instead of kmeans
>> TODO: try cluster lines and then find intersection point, but no kmeans, manually
