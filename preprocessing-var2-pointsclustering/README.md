# Preprocessing Variant `2: Cluster points`

## Description
Classical pipeline where we cluster points from lines not clustered 

## Pipeline
- [x] Gray level
- [x] Blur 3x3
- [x] Canny
- [x] Hough (rho and theta version)
- [ ] Cluster lines ***1**
- [x] Lines intersection
- [x] Cluster points ***2**
- [x] find 4 corners
- [x] geometric tran1sformations
- [ ] extract 64(8x8) sub-img  

## Comments and notes
> ****1***
>> Not clustering lines can make difficult to understand which points are 
>> not part of the grid, outliers points (because of the border or 
>> intersections outside the chessboard)