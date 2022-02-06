# AI_SPARK_CHALLENG_Object_Detection
제2회 연구개발특구 인공지능 경진대회 AI SPARK 챌린지

🏅 **Top 5% in mAP(0.75) (443명 중 13등, mAP: 0.98116)**

## 대회 설명
- **Edge 환경에서의 가축 Object Detection (Pig, Cow)**
- 실제 환경에서 활용가능한 Edge Device (ex: 젯슨 나노보드 등) 기반의 가벼운 경량화 모델을 개발하는 것이 목표이다.
- **가중치 파일의 용량은 100MB로 제한**한다.
- 가중치 파일의 용량이 100MB이하이면서 **mAP(IoU 0.75)를 기준으로 순위**를 매긴다.
- **본 대회의 모든 과정은 Colab Pro 환경에서 진행 및 재현한다.**

### Hardware
- **Colab Pro (P100 or T4)**

## Data
- **AI Hub에서 제공하는 가축 행동 영상 데이터셋 ([다운로드 링크](https://aihub.or.kr/aidata/30734/download))**
- [원천]소_bbox.zip: 소 image 파일
- [라벨]소_bbox.zip: 소 annotation 파일
- [원천]돼지_bbox.zip: 돼지 image 파일
- [라벨]돼지_bbox.zip: 돼지 annotation 파일
- 추가적으로, annotation에서의 "categories"의 값과 annotation list의 "category_id"는 소, 돼지 클래스와 무관하므로 이를 활용할 경우 잘못된 결과로 이어질 수 있다.

## Code
```
+- data (.gitignore) => zip파일만 최초 생성(AI Hub) 후 추가 데이터는 EDA 폴더 코드로부터 생성
|   +- [라벨]돼지_bbox.zip
|   +- [라벨]소_bbox.zip
|   +- [원천]돼지_bbox.zip
|   +- [원천]소_bbox.zip
|   +- Train_Dataset.tar (EDA - Make_Dataset_Multilabel.ipynb에서 생성) 
|   +- Valid_Dataset.tar (EDA - Make_Dataset_Multilabel.ipynb에서 생성)
|   +- Train_Dataset_Full.tar (EDA - Make_Dataset_Full.ipynb에서 생성)
|   +- Train_Dataset_mini.tar (EDA - Make_Dataset_Mini.ipynb에서 생성)
|   +- Valid_Dataset_mini.tar (EDA - Make_Dataset_Mini.ipynb에서 생성)
|   +- plus_image.tar (EDA - Data_Augmentation.ipynb에서 생성)
|   +- plus_lable.tar (EDA - Data_Augmentation.ipynb에서 생성)
+- data_test (.gitignore) => Inference시 사용할 test data (AI Hub으로부터 다운로드)
|   +- [원천]돼재_bbox.zip
|   +- [원천]소_bbox.zip
+- trained_model (.gitignore) => 학습 결과물 저장
|   +- m6_pretrained_full_b10_e20_hyp_tuning_v1_linear.pt
+- EDA
|   +- Data_Augmentation.ipynb (Plus Dataset 생성)
|   +- Data_Checking.ipynb (Error Analysis)
|   +- EDA.ipynb
|   +- Make_Dataset_Multilabel.ipynb (Train / Valid Dataset 생성)
|   +- Make_Dataset_Full.ipynb (Train + Valid Dataset 생성)
|   +- Make_Dataset_Mini.ipynb (Train mini / Valid mini Dataset 생성)
+- hyp
|   +- experiment_hyp_v1.yaml (최종 HyperParameter)
+- exp
|   +- hyp_train.py (본 코드와 같이 수정하여, 여러 실험 진행)
|   +- YOLOv5_hp_search_lr_momentum.ipynb (HyperParameter Tuning with mini dataset)
+- train
|   +- YOLOv5_ExpandDataset_hp_tune.ipynb (Plus Dataset을 활용하여 학습)
|   +- YOLOv5_FullDataset_hp_tune.ipynb (최종 결과물 생성)
|   +- YOLOv5_MultiLabelSplit.ipynb (초기 학습 코드)
+- YOLOv5_inference.ipynb
+- answer.csv (최종 정답 csv)
```
## Core Strategy
- **YOLOv5m6 Pretrained Model 사용 (68.3MB)**
- **MultiLabelStratified KFold (Box count, Class, Box Ratio, Box Size)**
- **HyperParameter Tuning (with GA Algorithm)**
- **Data Augmentation with Error Analysis**
- **Inference Tuning (IoU Threshold, Confidence Threshold)**

## EDA
<details>
    <summary>**자세히**</summary>

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

</details>
    
## Model
- YOLOv5m6 Pretrained Model 사용
- YOLOv5 계열 Pretrained Model 중 100MB 이하인 Model 선정 

||YOLOv5l Pretrained|YOLOv5m6 w/o Pretrained|YOLOv5m6 Pretrained|
|---|---|---|---|
|mAP@.5|0.9806|0.9756|0.9838|
|mAP@.5:.95|0.9002|0.8695|0.9156|

- 최종 사용 Model로서 YOLOv5m6 Pretrained Model 선택

## MultiLabelStratified KFold
- **PIG / COW의 Data의 개수에 대한 차이**
- **Image별 소유하는 Box의 개수에 대한 차이**
- 위 두 Label을 바탕으로 Stratified하게 Train/valid Split 진행

||Cow-Many|Cow-Medium|Cow-Little|Pig-Many|Pig-Medium|Pig-Little|
|---|---|---|---|---|---|---|
|Train|2739|1097|5886|2190|827|425|
|Valid|674|259|1497|559|221|81|

## HyperParameter Tuning
- Genetic Algorithm을 활용한 HyperParameter Tuning (YOLOv5 default 제공)
- Runtime의 제약(Colab Pro)으로 인한, Mini Dataset(50% 사용) 제작 및 HyperParameter Search 개별화 작업진행

### Core Code 수정
<details>
    <summary>**자세히**</summary>

```python
meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
        'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
        'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
        }

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3

        # Update할 HyperParameter만 new_hyp에 저장
        new_hyp = {}
        for k, v in hyp.items():
            if k in meta.keys():
                new_hyp[k] = v
        
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {save_dir}')  # download evolve.csv if exists

        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                # new_hyp에 있는 HyperParameter에 대해서만 meta값 불러오기
                g = np.array([meta[k][0] for k in new_hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    if k in new_hyp.keys(): # new_hyp에 존재하는 hyperParameter에 대해서만 Update
                        hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device, callbacks)
```

</details>
    
### Default HyperParameter vs Tuning HyperParameter
- obj, box, cls에 대한 HyperParameter에 따른 성능 변화폭 증가
(**NOTE: 학습 환경의 제약으로 인해, 각 성능비교표 마다 Epoch 수의 차이가 존재하여 성능의 차이가 있다. 성능 비교에만 참고하도록 하자**)

||Default|Tuning|
|---|---|---|
|obj_loss|0.023|0.003|
|box_loss|0.0095|0.0038|
|cls_loss|0.00003|0.00001|

||Default|Tuning|
|---|---|---|
|mAP@.5|0.9826|0.9824|
|mAP@.5:.95|0.8924|0.9016|

- Optimizer

||Adam|AdamW|SGD|
|---|---|---|---|
|mAP@.5|0.9635|0.9804|0.9848|
|mAP@.5:.95|0.8302|0.8994|0.914|

### 최종 변경 HyperParameter
|optimizer|lr_scheduler|lr0|lrf|momentum|weight_decay|warmup_epochs|warmup_momentum|warmup_bias_lr|box|cls|cls_pw|obj|obj_pw|iou_t|anchor_t|fl_gamma|hsv_h|hsv_s|hsv_v|degrees|translate|scale|shear|perspective|flipud|fliplr|mosaic|mixup|copy_paste|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|SGD|linear|0.009|0.08|0.94|0.001|0.11|0.77|0.0004|0.02|0.2|0.95|0.2|0.5|0.2|4.0|0.0|0.009|0.1|0.9|0.0|0.1|0.5|0.0|0.0|0.0095|0.1|1.0|0.0|0.0|

## Error Analysis
### 학습 결과 확인
|Data 양|Train|Valid|
|---|---|---|
|PIG|3442|881|
|COW|9722|2430|

|예측 결과|Label 개수|Precision|Recall|mAP@.5|mAP@.5:.95|
|---|---|---|---|---|---|
|PIG|3291|0.984|0.991|0.993|0.928|
|COW|3291|0.929|0.911|0.974|0.889|

- 위의 표와 같이, Cow의 Data의 양이 PIG의 Data보다 더 많다.
- YOLOv5 Pretrained Model의 경우 COCO Dataset에서 Cow 이미지를 보유하고 있다.
- 위의 두 가지 이점에도 불구하고, **Model이 Cow Detection에서의 어려움을 겪는다.**

### Box의 개수 및 Plotting
#### Box의 개수
![9](https://user-images.githubusercontent.com/53552847/152664271-afecf8e4-7987-4e12-bec8-126600e3ba28.PNG)

#### Train - Bounding Box Plotting
![10](https://user-images.githubusercontent.com/53552847/152664270-e6b4ec2f-6564-41fc-ae9f-f7b36cccc8a3.PNG)

#### Valid - Bounding Box Plotting
![11](https://user-images.githubusercontent.com/53552847/152664269-9d431af3-55d3-4931-bc17-21957b68f20d.PNG)

### Error 분석 결과
- 전반적으로 **Cow Dataset에서의 Bounding Box의 개수가 적다.**
- Image를 Plotting한 결과, **Cow Dataset에서의 Labeling이 제대로 되어있지 않다.**
    -  FP의 증가로 이어질 수 있다. (Labeling이 되어있지 않지만, Cow라고 예측)
-  이러한 결과로부터, Silver Dataset을 만들어 재학습시키도록 한다.
    - **학습된 Model로 Cow Image에 대하여 Bounding Box를 예측한다.**
    - **예측된 결과를 추가학습데이터로 활용한다.** 

## Data Augmentation with Silver Dataset
- YOLOv5m6 Pretrained with Full_Dataset(Train + Valid) (기존 Dataset으로 학습한 모델 활용)
- **총 12151개의 Cow Data에 대하여 Detection 진행 (IoU threshod: 0.7, Confidence threshold: 0.05)**

### Bounding Box 개수 시각화
![12](https://user-images.githubusercontent.com/53552847/152667315-7d2471fe-d363-49e4-a603-e7cde3fcb712.PNG)
- 위의 시각화자료로 부터, 분석가(본인)의 임의대로 **Bounding Box의 개수가 4개 이상인 Image만 최종 선정**
- **총 6628개의 Cow에 대한 Silver Dataset 추가**

## 결과
### 최종 선정 모델
- Dataset: Train + Valid Dataset을 학습
- YOLOv5m6 Pretrained Model 활용
- HyperParameter Tuning (위의 HyperParameter Tuning에서 작성한 표 참고)
- Inference Tuning (IoU Threshold: 0.68, Confidence Threshold: 0.001)

|Silver Dataset 결과비교|mAP@.75|
|---|---|
|최종 모델(w/o Silver Dataset)|0.98116|
|Plus Model(w Silver Dataset)|0.97965|

|Full vs Split 결과비교|mAP@.5|mAP@.5:.95|
|---|---|---|
|Full(Train + Valid)|0.9858|0.9271|
|Split(Train)|0.9845|0.9215|

## 시도했으나 아쉬웠던 점
### Knowledge Distillation
- 1 Stage Model to 1 Stage Model
- 성능이 높은 1 Stage Model을 찾으려고 했으나 YOLOv5x6을 적용하였을 때, mAP@.5: 0.9821 / mAP@.5:.95: 0.939로 점수의 큰 개선이 없었음.
- 즉, Teacher Model로 활용함으로서 얻어지는 이득이 적다.

## 회고
- Pretrained Model
    - COCO Dataset에서의 Cow Image의 형태는 어떠한지? 
    - Pig(COCO Dataset에 없음)의 경우, 잘 맞췄기 때문에 PreTrained Weight을 사용하지 않고 Epoch을 늘려서 학습하면 더 좋은 결과로 이어지지 않을까?
- Silver Dataset
    - Silver Dataset을 만드는 과정에 있어서, IoU Threshold와 Confidence Threshold를 최적화한다면 성능개선으로 이어질 수 있지 않을까?
    - Test Datsaet에서 애초에 Labeling이 제대로 되어있지 않는다면, 이러한 이유로 인해 필연적으로 성능개선이 안 이루어질 수 있지 않을까?
- MultiLabelStratified SPlit
    - Bounding Box와 Ratio와 Size에 따른 분류를 함께 진행해보면 어떨까?
    - 더불어, Bounding Box의 경우, Image가 가지고 있는 Box마다 다른데 이는 어떻게 MultiLabel하게 Split할 수 있을까?   
- 확실한 방법으로서 기존 Train Dataset에 Cow Image에 대한 Labeling을 직접했다면 성능 개선으로 이어지지 않았을까?!

## 추후 과제
- MultiLabelStratified Split 진행시, 각 이미지가 가지는 Bounding Box의 Ratio, Size에 따른 분류 방법 연구
- BackGround Image 넣기 => 탐지할 물체가 없는 Image를 추가해줌으로서 False Positive를 줄일 수 있다고 한다.
- 고도화된 HyperParameter Tuning 기법 적용 (ex, Bayesian Algorithm)
- Train Dataset에 대한 Silver Dataset을 만들어 이를 추가적으로 학습할 경우 성능 향상으로 이어지는지 알아보기 (Train Gold + Train Silver)
- Object Detection에서 SGD가 AdamW보다 좋은 것은 경험적인 결과인지 혹은 연구결과가 있는지 확인하기
- Pruning, Tensor Decomposition 적용해보기
- Object Detection Knowledge Distillation의 경우, 2 Stage to 1 Stage에 대한 방법론 찾아보기
