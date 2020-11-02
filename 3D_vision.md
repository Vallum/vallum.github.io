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
- 이상이 엔지니어를 위한 주로 2D, 3D rotation Group을 설명한 자료이고,
- 다음은 좀 더 근본적인 설명 문헌이다. 대수 전공자가 아닌 사람을 위해 쉽게 기초를 설명한다고 했는데, 대수학에 대한 충분한 지식없이 처음 봐서는 전혀 쉽지가 않다. 그냥 기록을 위해 남겨둔다.
- Very Basic Lie Theory, Roger Howe, vol90, pp600-623, The American Mathematical Monthly
- Naive Lie Theory, John Stillwell, 2008, Springer

## Multiple View Geometry

- 
