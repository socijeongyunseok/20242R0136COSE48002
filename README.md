CALIOPER Korean Version
===

Korean Context-Aware Offensive Language Detection - Focusing on Hate Speech toward Disability
<img width="636" alt="Model_Structure" src="https://github.com/user-attachments/assets/241ff0aa-1f9e-4c7a-949c-353a80f567e6" />

## 들어가며

본 연구는 맥락 기반 혐오 표현 탐지기의 개발을 통한 혐오를 조장하는 맥락에의 전산학적 접근을 목표로 한다. '언어의 본질적 특성에 기반하여 상해를 유발'하는 (Butler, 1997; Matsuda, 1993) 혐오 표현은 자연어처리(NLP)의 주요 연구 주제가 될 수밖에 없다. 혐오 표현은 '어떤 개인 혹은 집단에 대해 그들이 사회적 소수자의 속성을 가졌다는 이유로 차별·혐오하거나 차별·적의·폭력을 선동하는 표현'(국가인권위원회)이라는 정의상 발화가 이뤄지는 사회의 맥락에 큰 영향을 받을 수밖에 없다. 특히 일상어의 난점과 혐오 발화자들의 적극적 혐오 표현 감지 우회로 인해, 자연어 분류기의 기본적 형태인 문장 기반 분류 기법은 혐오 표현 분류에 있어 한계를 가질 수밖에 없다.

이에 본 연구는 혐오 표현의 분류에 있어 맥락 정보를 추가하여 학습하는 맥락 기반 혐오 표현 탐지를 도입하여 혐오 표현 탐지의 난점에 대응하고자 한다. 이를 위하여 본 연구는 챗봇 기반 혐오 표현 탐지기인 CALIOPER(Shin, 2023)를 개수하여 게시글-댓글 구조를 처리할 수 있도록 하고, 극단주의 사이트 FMKorea의 게시글과 댓글을 10만여 개를 크롤링하여 그 중 2만개를 선별, 영어 기반인 원본에 대응하는 한국어 데이터셋을 구축하며, 장애 혐오에 집중하여 CALIOPER 모델을 공격성, 장애 공격성, 장애 혐오와 이에 관한 멀티 데이터셋 모델로 개선한다. 전세계적 배외주의 열풍과 온라인 발 탈진실 현상의 강화가 두드러지는 우리 시대에, 본 연구는 혐오 표현에 대한 전산학적 대응 방안을 제시함으로써 사회 공동체의 증오를 막고 존엄성을 지키는 데에 기여하고자 한다.

## 모델 소개

**주요 특징**:

* **맥락 기반 탐지**: 단일 발화뿐만 아니라 이전 발화를 고려하여 발언의 맥락을 이해.
* **다양한 레이블 적용**: 장애 혐오와 관련된 다중 레이블(예: 'offensiveness\_disabled', 'hate\_speech\_disabled')을 포함.

* **CALIOPER**:

  * 사전 학습된 BERT 및 Sentence-BERT 기반.
  * Attention 메커니즘을 활용하여 맥락 정보를 통합.
  * 장애 혐오와 같은 세부적인 혐오 발언 탐지 가능.

* **맞춤 손실 함수**:

  * 주요 혐오와 맥락 의존적 혐오를 결합한 손실 계산.
 
* **한국어 데이터셋**
* 본 연구는 혐오 데이터셋을 확보하기 위하여 FM코리아 게시글 약 12만 건을 크롤링하여, 게시글과 댓글의 위계 구조를 보존한 채로 previous utterance를 형성하였으며, 이 중 2만 건을 선정 6개의 다중 레이블을 활용하게 하였다.

## 기존 모델과의 비교 요약

| 항목                    | 기존 CALIOPER               | 개선된 모델                          |
|-------------------------|-----------------------------|--------------------------------------|
| 언어 지원               | 영어 중심                  | 한국어 중심(kcbert 등)                |
| 문장 임베딩 방식        | [CLS] 토큰                | Mean Pooling/Trainable Attention 적용 |
| 맥락 정보 처리 방식     | 단순 Attention            | Key/Value Transform Attention, Caching|
| 멀티레이블 확장         | 제한적                     | 멀티레이블 분류 기반 확장 용이        |
| 평가 전략               | 단순 Train/Test 분리       | K-Fold 검증, 다양한 메트릭(F1, dep-F1)|
| 결과 관리               | 단순 저장                  | 명확한 파일명 규칙, CSV 분석 결과 기록 |

---

## 최종 F1 Score

| Fold | Valid Micro F1 | Valid Macro F1 | Best Valid Micro F1 | Best Valid Macro F1 |
|------|----------------------|---------------------|---------------------------|---------------------------|
| 1    | 0.7465               | 0.7043              | 0.8329                    | 0.7904                    |
| 2    | 0.7298               | 0.7002              | 0.8286                    | 0.7886                    |
| 3    | 0.7312               | 0.6481              | 0.8266                    | 0.7866                    |
| 4    | 0.7474               | 0.7121              | 0.8308                    | 0.7845                    |
| 5    | 0.7274               | 0.6951              | 0.8304                    | 0.7898                    |

cf. Original CALIOPER의 Dep-F1는 70.1 

---

## 각 레이블 별 성능

| Category                         | Precision | Recall | F1-Score | Support |
|----------------------------------|-----------|--------|----------|---------|
| offensive                        | 0.8       | 0.834  | 0.818    | 842.2   |
| context_dependent                | 0.906     | 1      | 0.95     | 2211.2  |
| offensiveness_disabled           | 0.778     | 0.8    | 0.788    | 1093    |
| context_offensiveness_disabled   | 0.748     | 0.84   | 0.79     | 1207.2  |
| hate_speech_disabled             | 0.686     | 0.694  | 0.692    | 404.4   |
| context_hate_speech_disabled     | 0.618     | 0.794  | 0.692    | 729.6   |

| Category                         | Precision | Recall | F1-Score | Support |
|----------------------------------|-----------|--------|----------|---------|
| micro_avg                        | 0.79      | 0.872  | 0.83     | 6487.6  |
| macro_avg                        | 0.756     | 0.828  | 0.788    | 6487.6  |
| weighted_avg                     | 0.796     | 0.872  | 0.83     | 6487.6  |
| samples_avg                      | 0.778     | 0.808  | 0.758    | 6487.6  |

## 데이터셋

* **데이터 구조**:

  * `train_not_U.tsv`, `test_not_U.tsv`: 불확실한 혐오 레이블을 제외한 학습 및 테스트 데이터.
  * 다중 턴 대화 데이터에서 장애 혐오 탐지를 위해 `previous_utterance`, `context_offensiveness_disabled` 등 다양한 정보를 포함.


## 설치 및 사용법
   - 실행 전 미리 data 폴더를 main.py와 같은 폴더에 배치
1. **설치**
```
 git clone https://github.com/socijeongyunseok/20242R0136COSE48002.git
 pip install -r requirements.txt
```
2. **명령행**
```
python main.py --train\_data fm\_yunseok --test\_data fm\_yunseok --main\_encoder beomi/kcbert-large --context\_encoder beomi/kcbert-large --batch\_size 32 --gpus 0 --repeat 5 --epochs 20 --mean\_pooling
```

## 일러두기
본 연구는 CALIOPER(Mingi shin, 2023)의 연구에 기반하여 만들어졌음.
- https://github.com/mingi-sid/CALIOPER
