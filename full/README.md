# Preprocessing Variant `Some tries and first approach`

## Description
Started with classical pipeline for grid detection

## Pipeline
- [x] Gray level
- [ ] Blur 3x3

### First pass
- [x] Canny ***1**
- [ ] Hough Standard (rho and theta version)
- [x] Hough Probabilistic (2 points version) ***2**
- [x] Clean lines outliers
- [x] Cluster lines: DBSCAN ***3**
- [x] find 4 corners
- [x] geometric tran1sformations
### Second pass
- [x] Canny
- [x] Hough Probabilistic (2 points version)
- [x] Clean lines outliers
- [x] Cluster lines: DBSCAN
- [x] Remove outliers: borders, Add missing lines
- [x] Find points by Lines intersection ***4**
- [x] Extract 64(8x8) squares sub-img

## Comments and notes
> ****1***
>> No more auto-canny <br>
>> With new dataset with recurrent resolution, lights etc we fixed parameters: <br>
>> low: 70/90; high: 400; aperture: 3. 

> ****2***
>> Tried Standard and probabilistic <br>
>> Std thresh=90. <br>
>> Prob thresh=75, minLineLength=30, maxLineGap=50 <br>
>> Probabilistic is solid.

> ****3***
>> Tried different approaches <br>
>> Until now: manual, kmeans, no clustering -> cluster points later, DBSCAN <br>
>> DBSCAN works best

> ****4***
>> no need to cluster points if we have clustered lines
