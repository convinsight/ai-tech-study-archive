# 3D Understanding

#### 1.   Seeing the world in 3D perspective

1.  Why is 3D important?
   - 우리가  3D 세상에 살고 있기 때문
   - AI를 제대로 활용하기 위해 3D에 대한 이해가 바탕이 되어야 함
   - 활용
     - AR/VR
     - 3D printing
     - medical/chemical
2. The way we observe 3D
   - Triangulation
     - Two-view geometry
     - n-view로 확장 가능
       - 교재: Multiple View Geometry 참고
3. 3D data representation
   - Multi-view image
   - Voilumetric(voxel)
   - Part assembly
   - Point cloud
   - Mesh(Graph CNN)
   - Implicit shape
4.  3D datasets
   - ShapeNet
     - 55000
     - 3D 데이터셋이 희귀한 상황에서 나름 거대한 스케일
   - PartNet
     - 26000
     - 573585 part instances
   - SceneNet
     - 5 million RGB-Depth
     - indoor images
   - ScanNet
     - 1500 scan
   - KITTY, Semantic KITTI, Waymo Open Dataset
     - oudoor 3D scene datasets
     - 자율주행을 염두에 둔 데이터 셋

#### 2. 3D tasks

1.  3D recognition
   - 3D Model
     - e.g. Volumetric CNN
2. 3D object detection
   - 3D bounding box
   - 자율주행에서 유용하게 사용
2. 3D semantic segmentation
   - 물체를 구분
   - 물체의 부분을 구분
4.  Conditional 3D generation
   - Mesh R-CNN
     - Input: 2D image
     - output: 3D meshes
   - Decomposing
     - multiple sub-problem으로 재구성
     - 물리적으로 의미있는 단위로 분리



#### 3. Example

1. Photo refocussing
   - post-refocusing
   - portrait mode
   - depth sensor나 NN을 통해 측정된 depth map을 사용
   - Procedures
     1. depth threshold range설정
        - 값 외부의 focus 날림
     2. range 내부와 외부 두 개의 mask 생성
     3. input image blurr버전 생성
     4. masked focus/defocus image 생성
     5. 두 이미지를 잘 섞어서 만듦



