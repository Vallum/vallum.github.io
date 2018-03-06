This is the summary of VggFace2 https://arxiv.org/pdf/1710.08092.pdf

# Abstract
- 3.31 million images
- 9131 subjects
- average 362.6 images (min 80, max 800)
- train with ResNet-50 with/out Squeeze-and-Excitation
- with CNN on VGGFace2 on MSCeleb-1M and on union of them
- SOTA on  IJB-A and IJB-B by a large margin

# Introduction
* four contributions
  - new large scale dataset
  - multiple stages of automatic and manual filtering to minimise label noise
  - template annotaion for pose and age recognition performance
  - SOTA on IJB-A and IJB-B

# DataSets
* LFW (Labelled Faces in the Wild, 2007)
  - 13,000 images
  - 5749 identities

* CelebFaces+ (2014)
  - 202,599 images
  - 10,177 celebs

* CASIA-WebFace (2014)
  - 494,414 images
  - 10,575 people
  - training only

* FaceScrub (2014)
  - 4,000 images
  - 80 identities
  - evaluation

* FG-NET (2014)
  - 975 images
  - 82 ientities
  - evaluation

* VGGFace (2015)
  - 2.6 mio images
  - 2622 people
  - the curated 800,000 images and 305 images per id (noise is removed by human annotators)
  - training only

* MegaFace (2016)
  - 4.7 mio images
  - 672,057 identities
  - distractors for evaluation with FaceScrub and FG-NET

* Ms-Celeb-1M (2016)
  - 10 mio images
  - 100k celebs
  - average of 81 images per person
  - intra-identity variation is restricted
  - label noise

* UMDFaces (2016)
  - 367,920 images
  - 8,501 identities
  - average 43.3 identities

* IJB-A (IARPA Janus Benchmark-A, 2015)
* IJB-B (IARPA Janus Benchmark-B, 2017)
  - evaluation

* YTF (Youtube Face, 2011)
  - 3,425 videos
  - 1,595 identities

* UMBFaces-Videos
  - 22,075 videos
  - 3,107 identities

* Facebook
  - 500 mio images
  - 10 mio subjects

* Google
  - 200 mio images
  - 8 mio identities

# VGGFaces Overview
* Stats
  - 59.7% male
  - min 87, max 843 images for each identities
  - bounding boxes
  - 5 keypoints
* Split
  - training 8631 classes
  - evaluation 500 classes
* Pose and Age Annotations
  - Pose template : 5 images for 3 view, 1.8k templates
  - Age template : 5 images for below 34 or above, 400 templages
  
# Dataset Collection
  - Detecting overlapped subjects
  - Removing outlier images for a subject
  
# Experiments
* Architecture
  - Resnet-50
  - SE-Resnet-50
  
  