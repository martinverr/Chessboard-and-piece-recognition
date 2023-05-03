# Preprocessing Variant `Some tries and first approach`

## Description
Started with the main version pipeline, tried Laplacian to compare it with Canny

## Pipeline
- [x] Gray level
- [x] Blur 3x3
- [x] Canny, Laplacian, bilateral (indipendently) ***1**
- [x] Hough (rho and theta version) 

## Comments and notes
> ****1***
>> Canny the best result for everyone
>> Bilataral best for some images, bad for few
>> Laplacian worst (binarize it leave to bad result, but needed for houghlines)