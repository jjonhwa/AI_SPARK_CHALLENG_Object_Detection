# AI_SPARK_CHALLENG_Object_Detection
제2회 연구개발특구 인공지능 경진대회 AI SPARK 챌린지

🏅 **Top 5% in mAP(0.75) (225팀 중 13등, mAP: 0.98116)**

## 대회 설명
- Edge 환경에서의 가축 Object Detection (Pig, Cow)
- 실제 환경에서 활용가능한 Edge Device (ex: 젯슨 나노보드 등) 기반의 가벼운 경량화 모델을 개발하는 것이 목표이다.
- 가중치 파일의 용량은 100MB로 제한한다.
- 가중치 파일의 용량이 100MB이하이면서 mAP(IoU 0.75)를 기준으로 순위를 매긴다.
- **본 대회의 모든 과정은 Colab Pro 환경에서 진행 및 재현한다.**

## Data
- AI Hub에서 제공하는 가축 행동 영상 데이터셋 ([다운로드 링크](https://aihub.or.kr/aidata/30734/download))
- [원천]소_bbox.zip: 소 image 파일
- [라벨]소_bbox.zip: 소 annotation 파일
- [원천]돼지_bbox.zip: 돼지 image 파일
- [라벨]돼지_bbox.zip: 돼지 annotation 파일
- 추가적으로, annotation에서의 "categories"의 값과 annotation list의 "category_id"는 소, 돼지 클래스와 무관하므로 이를 활용할 경우 잘못된 결과로 이어질 수 있다.

## Core Strategy
- YOLOv5m6 Pretrained Model 사용 (68.3MB)
- MultiLabelStratified KFold (Box count, Class, Box Ratio, Box Size)
- HyperParameter Tuning (with evolve)
- Data Augmentation with Error Analysis
