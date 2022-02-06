# AI_SPARK_CHALLENG_Object_Detection
제2회 연구개발특구 인공지능 경진대회 AI SPARK 챌린지

🏅 **Top 5% in mAP(0.75) (225팀 중 13등, mAP: 0.98116)**

## 대회 설명
- **Edge 환경에서의 가축 Object Detection (Pig, Cow)**
- 실제 환경에서 활용가능한 Edge Device (ex: 젯슨 나노보드 등) 기반의 가벼운 경량화 모델을 개발하는 것이 목표이다.
- **가중치 파일의 용량은 100MB로 제한**한다.
- 가중치 파일의 용량이 100MB이하이면서 **mAP(IoU 0.75)를 기준으로 순위**를 매긴다.
- **본 대회의 모든 과정은 Colab Pro 환경에서 진행 및 재현한다.**

## Data
- **AI Hub에서 제공하는 가축 행동 영상 데이터셋 ([다운로드 링크](https://aihub.or.kr/aidata/30734/download))**
- [원천]소_bbox.zip: 소 image 파일
- [라벨]소_bbox.zip: 소 annotation 파일
- [원천]돼지_bbox.zip: 돼지 image 파일
- [라벨]돼지_bbox.zip: 돼지 annotation 파일
- 추가적으로, annotation에서의 "categories"의 값과 annotation list의 "category_id"는 소, 돼지 클래스와 무관하므로 이를 활용할 경우 잘못된 결과로 이어질 수 있다.

## Core Strategy
- **YOLOv5m6 Pretrained Model 사용 (68.3MB)**
- **MultiLabelStratified KFold (Box count, Class, Box Ratio, Box Size)**
- **HyperParameter Tuning (with GA Algorithm)**
- **Data Augmentation with Error Analysis**

## EDA
### Cow Dataset vs Pig dataset
||PIG|COW|
|---|---|---|
|Image 개수|4303|12152|
- Data의 분포가 "Cow : Pig = 3 : 1"
- **Train / Valid split할 경우, 골고루 분포하도록 진행**
    
### Image size 분포
||Pig Image Size|Cow Image Size|
|---|---|---|
|1920x1080|3131|12152|
|1280x960|1172|0|
- 대부분의 Image의 크기는 1920x1080
- Pig Data에서 일부 image의 크기가 1280x960
- **좌표변환 적용시, Image size를 고려하여 변환**
    
### Box의 개수에 따른 분포
![3](https://user-images.githubusercontent.com/53552847/152643870-b34f9ba1-7921-4aae-ad7d-1777d2d819ae.PNG)
- pig data와 cow data에서 Box의 개수가 서로 상이하게 분포
- **Train / Valid split할 경우, 각 image별로 가지는 Box의 개수에 따라서 골고루 분포할 수 있도록 진행.**
    
### Box의 비율에 따른 분포
![4](https://user-images.githubusercontent.com/53552847/152643869-7cae1b57-88f4-42f8-a672-4c6fc52ec58a.PNG)
- pig data와 cow data에서 Box의 비율은 유사
- **Train / Valid split할 경우, 각 image별로 가지는 Box의 비율에 따라서 골고루 분포할 수 있도록 진행.**
    
### Box의 크기에 따른 분포
![5](https://user-images.githubusercontent.com/53552847/152643868-60c61f6e-e9b4-478b-9214-2c07199bf2be.PNG)
- pig data, cow data 모두 small size bounding box (넓이: 1000~10000)의 개수가 상대적으로 적음.
- **small size bounding box를 지울 것인가? => 선택의 문제 (본 과정에서는 지우지 않음)**

### Small size bounding box에 대한 세밀한 분포 조사
![6](https://user-images.githubusercontent.com/53552847/152643866-7a4fea1d-6901-4bb1-b8bd-b0dedadf5ef2.PNG)
|넓이가 4000이하인 Data의 개수|PIG|COW|
|---|---|---|
|개수|137|71|
|비율|0.003|0.0018|
- 넓이가 4000이하인 Data의 개수가 pig data 137개, cow data 71개
- 전체 Data에 대한 비율 (137 -> 0.003, 71 -> 0.0018). 즉, 0.3%, 0.18%
- **넓이가 4000이하인 Bounding Box를 지울 것인가? => 선택의 문제 (본 과정에서는 지우지 않음)**

### Box가 없는 이미지 분포
|Box가 없는 이미지|PIG|COW|
|---|---|---|
|개수|0|3|
- Cow Image에서 3개 존재
- White Noise로 판단하여 삭제하지 않음.

## Model
- YOLOv5m6 Pretrained Model 사용
- YOLOv5 계열 Pretrained Model 중 100MB 이하인 Model 선정 

||YOLOv5l Pretrained|YOLOv5m6 w/o Pretrained|YOLOv5m6 Pretrained|
|---|---|---|---|
|mAP@.5|0.9806|0.9756|0.9838|
|mAP@.5:.95|0.9002|0.8695|0.9156|

- 최종 사용 Model로서 YOLOv5m6 Pretrained Model 선택

## MultiLabelStratified KFold
- PIG / COW의 Data의 개수에 대한 차이
- Image별 소유하는 Box의 개수에 대한 차이
- 위 두 Label을 바탕으로 Stratified하게 Train/valid Split 진행

||Cow-Many|Cow-Medium|Cow-Little|Pig-Many|Pig-Medium|Pig-Little|
|---|---|---|---|---|---|---|
|Train|2739|1097|5886|2190|827|425|
|Valid|674|259|1497|559|221|81|

## HyperParameter Tuning
- Genetic Algorithm을 활용한 HyperParameter Tuning (YOLOv5 default 제공)
- Runtime의 제약(Colab Pro)으로 인한, Mini Dataset(50% 사용) 제작 및 HyperParameter Search 개별화 작업진행
++HyperParameter Search 개별화 코드 삽입++
++이전 HyperParameter와 이후 HyperParameter 비교++
++Adam, AdamW, SGD 비교표 작성++

## Error Analysis
### 학습 결과 확인
|Data 양|Train|Valid|
|PIG|3442|881|
|COW|9722|2430|

|예측 결과|Label 개수|Precision|Recall|mAP@.5|mAP@.5:.95|
|PIG|3291|0.984|0.991|0.993|0.928|
|COW|3291|0.929|0.911|0.974|0.889|

- 위의 표와 같이, Cow의 Data의 양이 PIG의 Data보다 더 많다.
- YOLOv5 Pretrained Model의 경우 COCO Dataset에서 Cow 이미지를 보유하고 있다.
- 위의 두 가지 이점에도 불구하고, Model이 Cow Detection에서의 어려움을 겪는다.

### Box의 개수 및 Plotting
#### Box의 개수
![9](https://user-images.githubusercontent.com/53552847/152664271-afecf8e4-7987-4e12-bec8-126600e3ba28.PNG)

#### Train - Bounding Box Plotting
![10](https://user-images.githubusercontent.com/53552847/152664270-e6b4ec2f-6564-41fc-ae9f-f7b36cccc8a3.PNG)

#### Valid - Bounding Box Plotting
![11](https://user-images.githubusercontent.com/53552847/152664269-9d431af3-55d3-4931-bc17-21957b68f20d.PNG)

### Error 분셕 결과
- 전반적으로 Cow Dataset에서의 Bounding Box의 개수가 적다.
- Image를 Plotting한 결과, Cow Dataset에서의 Labeling이 제대로 되어있지 않다.
    -  FP의 증가로 이어질 수 있다. (Labeling이 되어있지 않지만, Cow라고 예측)
-  이러한 결과로부터, Silver Dataset을 만들어 재학습시키도록 한다.
    - 학습된 Model로 Cow Image에 대하여 Bounding Box를 예측한다.
    - 예측된 결과를 추가학습데이터로 활용한다.    

## Data Augmentation with Error Analysis
++Cow Dataset Augmentation++

## 결과
++Full Dataset 활용++
++Tuning으로 찾은 HyperParameter 활용++
++Augmentation Model vs Non Augmentation Model++
++Non Augmentation Model 활용++
++Inference Tuning++
++결과값 비교 표 작성++

## 회고
- COCO Dataset에 Cow Image는 우리 Dataset과 상이하지 않을까?
- Pretrained Weight를 사용하지 않고 Epoch을 늘려 학습했다면 더 좋은 결과로 이어지지 않을까?
- 
++Plus Dataset에서 Confidence Threshold가 0.001++

## 추후 과제
- MultiLabelStratified Split 진행시, 각 이미지가 가지는 Bounding Box의 Ratio, Size에 따른 분류도 함께 진행하기
- BackGround Image 넣기 => 탐지할 물체가 없는 Image를 추가해줌으로서 False Positive를 줄일 수 있다고 한다.
- 고도화된 HyperParameter Tuning 기법 적용 (ex, Bayesian Algorithm)
- Silver Dataset을 활용할 때, 중복되는 Data에 대하여 Bounding Box만 추가할 경우, 학습 성능의 차이 비교
