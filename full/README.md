# Preprocessing Variant `Some tries and first approach`

## Description
Started with classical pipeline for grid detection

## Pipeline
- [x] Gray level
- [ ] Blur 3x3
- [x] Canny ***1**
- [ ] Hough Standard (rho and theta version)
- [x] Hough Probabilistic (2 points version) ***2**
- [x] Clean lines outliers
- [x] Cluster lines: DBSCAN ****3**
- [x] Lines intersection
- [ ] Cluster points ***4**
- [x] find 4 corners
- [x] geometric tran1sformations
- [ ] extract 64(8x8) sub-img  

## Comments and notes
> ****1***
>> No more auto-canny
>> With new dataset with recurrent resolution, lights etc we fixed parameters:
>> low: 70/90; high: 400; aperture: 3. 

> ****2***
>> Tried Standard and probabilistic
>> Std thresh=90.
>> Prob thresh=75, minLineLength=30, maxLineGap=50
>> Probabilistic is solid.

> ****3***
>> Tried different approaches
>> Until now: manual, kmeans, no clustering -> cluster points later, DBSCAN
>> DBSCAN works best

> ****4***
>> no need to cluster points if we have clustered lines