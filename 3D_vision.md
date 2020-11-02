# Reconstruction from Multiple Images

## Why?

- RGB bitmap에서 2D convolution 정보를 통해서 visual feature를 추출하여 비교, 해석, 이용하는 일이 대세가 되어 있다.
- 그러나, 그러한 경우에 visual feature라는 것은 보통 contour, edge, texture가 혼합된 정보를 필터 맵 같은것에 저장하여 이용하는 것을 의미한다.
- 따라서, 어쩔 수 없이 이미지 내의 오브젝트(매니폴드 혹은 폴리곤)의 원래 성질을 파악하거나 이용하는 것은 제한적이 될 수 밖에 없다.
- 즉, 극단적으로 말해서 실제 카메라로 찍은 사진에서 충분히 많은 정보를 이용하지 못하고, 이미지에 담긴 "느낌적인 느낌"만 이용하는데에 머무른다는 느낌이다.
- 실제 카메라에서 얻은 이미지의 경우에 최대한 많은 실제 오브젝트의 정보를 추출해 낼 수 없을까?
- 가령, 사진에 담긴 오브젝트간의 공간적 거리감 같은 것을 재현하게 되면, edge contour detection같이 애매한 방법으로 object detection을 하는 대신에
- 좀 더 자연스럽게 오브젝트들을 분리 할 수 있지 않을까?

## Backgrounds

- 일견, 단순해 보이는 미션으로 2D Image에 depth란 한 차원을 추가하여 3D 정보로 재구성하는 미션이 있다.
  - 예를 들면, 비디오 스트림 등에서 Depth를 추정하는 연구들이 있다.
  - Learning Depth from Monocular Videos using Direct Methods, 2018, Chaoyang Wang, Jose Miguel Buenaposada, Rui Zhu, Simon Lucey
  - Deeper Depth Prediction with Fully Convolutional Residual Networks, 2016, Iro Laina, Christian Rupprecht, Vasileios Belagiannis
  - Unsupervised Learning of Depth and Ego-Motion from Video, 2017, Tinghui Zhou, Matthew Brown, Noah Snavely, David G. Lowe
  - Semi-Supervised Deep Learning for Monocular Depth Map Prediction, 2017, Yevhen Kuznietsov, Jorg Stuckler, Bastian Leibe
- Depth를 추정한다고 하지만, 사실은 카메라 Pose를 추정하고 카메라 위치에서의 이미지 각 픽셀까지의 상대 거리를 재구성하는 것으로 보인다.

- 근본적으로 이와 같은 미션을 Visual Odometry라고 하는 것 같다.
  - "In robotics and computer vision, visual odometry is the process of determining the position and orientation of a robot by analyzing the associated camera images. It has been used in a wide variety of robotic applications, such as on the Mars Exploration Rovers."
  - https://en.wikipedia.org/wiki/Visual_odometry
  - 이미지에 담겨 있는 암묵적인 기하 정보를 추정(Estimation)하는게 목적인 것이다.
- 이를 이용하는 좀 더 고차원적인 미션으로 SLAM이란 것이 있다.
  - simultaneous localization and mapping (SLAM) 
  - 카메라 등을 통해서 들어오는 시각 정보로, 3차원적인 지리 정보를 재구성하는 것이다.
  - 예를 들면, 지도 만드는 로봇 혹은 자율주행 같은 미션이 해당된다.
  - 자율주행을 테스트 하기 위한 KITTI라는 dataset이 있다.
  - http://www.cvlibs.net/datasets/kitti/

## Direct Visual Odometry

- 2차원 이미지에서 어떻게 3차원 정보를 복원하는가?
- Direct Visual Odemetry라고 부르는데, 일종의 현대수학을 이용한 3각 측량법이다.
- Real-Time Visual Odometry from Dense RGB-D Images, 2011, Frank Steinbrücker, Jürgen Sturm, Daniel Cremers
- Lie Theory를 이용한다. 
- 간단히 말해서, 3차원 세상을 직접 행렬로 모델링하면 3-by-3이라서 9개의 파라미터에 대한 최적화가 필요하다.
- 그런데, 카메라는 실제로는 Rigid Body Motion이라는 물리적 제약을 받는 운동만 가능하다는 점을 고려하여, 
- Rotation과 Translation이라는 모션만 감안하여 이를 Tangent Space에서 모델링하게 되면, 파라미터는 5개로 줄어든다.
- 그리고 Vector Space는 Tangent Space이기 때문에 선형대수의 여러 성질을 활용할 수 있게 된다.
- Lie Theory가 나오는 이유는,
- 원래 카메라의 관점은 Lie Group으로 표현되는데, Rigid Body Motion의 제약을 반영하여 Lie Algebra라는 좀 더 계산이 편리한 Vector space로 변형하는 과정이 바로 Lie 이론이다.

## Lie Theory

- Lie 이론은 매우 추상적인 현대 수학의 대수학 이론이기 때문에, 관련 전공 수학자가 아니라면 접근이 매우 어렵다.
- 그러나, 매우 높은 추상성 덕분에 현실의 매우 어려운 문제를 올바르게 표현할 수 있는 강력한 힘이 있다는 것이 발견되어,
- 예로 부터(?), 이론 물리학과 양자역학 등을 제대로 이해하는데 도움을 주는 도구로 종종 여겨졌고, Geometry의 문제가 관여되는 현대 과학기술 분야에서 그 쓰임새가 매우 큰 것으로 재발견되고 있다.
- 따라서, Lie 이론을 어떻게 과학자들에게 설명할 수 있는가가 하나의 큰 문제가 되어 있다.
- 다음과 같은 문헌들을 참고할 수 있었다.
- A micro Lie theory for state estimation in robotics, 2018, Joan Sola, Jeremie Deray, Dinesh Atchuthan, arXiv 1812.01537
- Lie Groups for 2D and 3D Transformations, 2017, Ethan Eade, Technical Report
- Lie groups, Lie algebras, projective geometry and optimization for 3D Geometry, Engineering and Computer Vision, 2014, Tom Drummond, Lecture Notes
- Lie Groups and Lie Algebras in Robotics, 2006, Jonathan Selig
- 이상이 엔지니어를 위한 주로 2D, 3D rotation Group을 설명한 자료이고,
- 다음은 좀 더 근본적인 설명 문헌이다. 대수 전공자가 아닌 사람을 위해 쉽게 기초를 설명한다고 했는데, 대수학에 대한 충분한 지식없이 처음 봐서는 전혀 쉽지가 않다. 그냥 기록을 위해 남겨둔다.
- Very Basic Lie Theory, Roger Howe, vol90, pp600-623, The American Mathematical Monthly
- Naive Lie Theory, John Stillwell, 2008, Springer

## 3D Reconstruction

- 소스 형태가 동영상이 되었든,
- 근본적으로는 두 장 혹은 여러 장의 이미지에서 이미지에 담긴 3차원 정보를 복원하는 것을 목적으로 하고 있다.
- SLAM 같은 경우가 그 일부에서 혹은 그에 바탕하여 확장되어 설정된 특정한 미션이다.
- 요즘 같은 경우에 당연히 Deep Learning과 결합하여 효과적인 연구 결과물이 나올 것이라고 예상해볼 수 있다.
- CVPR 2020의 Deep Direct Visual SLAM에 대한 Daniel Cremers교수의 키노트를 소개해본다.
- https://sites.google.com/view/omnicv-cvpr2020/program/keynote-daniel-cremers
- 그러나 기본적으로 Direct Visual Odometry를 확장하는 개념이다.
- 관련 전공자들은 어떤 문헌 자료를 통해서 이 분야에 처음 접근하는게 좋다고 생각하는가 찾아보았다.
- https://www.reddit.com/r/computervision/comments/ceu057/best_books_and_courses_on_visual_odometry_and_3d/
```
Geoe0
I can recommand SLAM for mobile robotics: https://www.amazon.de/dp/1466621044/?coliid=I2IP24301C8HSY&colid=1N3R5MT3K6FKJ&psc=1&ref_=lv_ov_lig_dp_it 
Its not for aerial robots but its really good book for SLAM. 
For aerial specifically I would suggest the works for Cremers et al. from the TU München. 
There is also a Maters Thesis from one of his students about Dense VO. 
Its very well written. 
Also the research of Scaramuzza from the ETH Zürich is very good. 
His tutorial paper is a good starting point https://www.ifi.uzh.ch/dam/jcr:5759a719-55db-4930-8051-4cc534f812b1/VO_Part_I_Scaramuzza.pdf

alkasm
Just to tack on here for stuff from TUM, Prof Cremers has a course on Multiple View Geometry and Dr. Sturm has a course on Visual Navigation for Flying Robots, both fully on YouTube. 
They are both excellent resources and start without a lot of assumptions about people's background. 
I mean in general just subscribe to the channel and check out their other courses if you're interested.
```
- 우연이 아닌 것으로 보이지만, 추천된 Cremers와 Scaramuzza는 둘 다 CVPR 2020의 Keynote speaker이다.

## Multiple View Geometry
- 유튜브에 뮨헨 공과 대학의 Daniel Cremers의 Multiple View Geomery 강의가 있다. 
- https://www.youtube.com/playlist?list=PLTBdjV_4f-EJn6udZ34tht9EVIW7lbeo4
- 요약을 하자면, 고급(?) 선형대수의 이론에서 Rigid Body Motion에 대한 제약이 걸린 SO(3) group에 대해서만 Lie Theory를 살짝 적용한 뒤에,
- 실제로 multiple image source에서 3D reconsruction을 하기 위해 필요한 이론을 설명한다.
- 매우 쉽게 설명한다고 생각되지만, 사실 Singular Value Decomposition이나 Linear Transformation, Spectral Theory 등의 선형대수 지식은 매우 잘 알고 있다는 전제하에 설명한다.
- 즉, Lie Theory까지 이해할 필요는 없지만 선형대수는 매우 깊고 넓게 이해하고 있어야 한다.
- 3D vision 분야가 그렇다고 생각된다.
- 강의 슬라이드 및 공식페이지는 https://vision.in.tum.de/teaching/online/mvg
- Daniel Cremers 강의의 교재는
- "An Invitation to 3D Vision", 2004, Ma, Soatto, Kosecka, Sastry
- 그리고, 강의에서도 소개되지만, 널리 알려진 3D Vision 및 Multiple View Geometry 교재는 다음과 같은 것들이 있다.
- "The Geometry of Multiple Images", 2001, Faugeras and Luong
- "Multiple View Geometry", 2003, Hartly and Zisserman
- 저자들이 보기에 교재 내용이 별로 이론적으로는 업데이트 할만한 내용은 없다고 생각하는것 같다. 나온지 오래되었다.
- 다시 말하면 이 정도는 기본적으로 알아야 된다는 것으로 보인다.
- 에피폴라 기하학 같은 것을 들어봤다면, 바로 여기서 설명이 된다.
- 여러장의 이미지에서 3D 정보를 추정한다는 것은 결국 Essential Matrix에 대한 Estimation으로 요약된다.
- https://www.researchgate.net/publication/220182618_Some_Properties_of_the_E_Matrix_in_Two-View_Motion_Estimation
- 혹은 Camera calibration 정보가 없을 경우에는 E-Matrix 대신에 F-Matrix, Fundamental Matrix라고 표현하는 것 같다.
- On determining the fundamental matrix : analysis of different methods and experimental results, 2012, Quang-Tuan Luong, Rachid Deriche, Olivier Faugeras, Théodore Papadopoulo, hal.inria.fr

## Deep Direct Visual Odometry
- Deep Direct Visual Slam : https://vision.in.tum.de/research/vslam/d3vo
- DDVO : https://arxiv.org/pdf/1912.05101.pdf

## Reference Keywords
- Visual Odemetry
- simultaneous localization and mapping (SLAM)
  - drift error
- structure-from-motion (SFM)
- Lucas-Kanade Algorithm
- Epipolar geometry, epipolar plane, epipolar constraint
- bundle adjustment
- optical flow
