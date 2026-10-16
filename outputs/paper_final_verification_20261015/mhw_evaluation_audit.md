# SST·MHW 평가 정의와 기준 재검산

작성: 2026-10-15 13:59 KST. 신규 학습·재튜닝·원본 코드/데이터/기존 결과 변경 없음.

**PASS WITH LIMITATIONS. 기존정의를그대로검산했으며threshold를수정하지않았다.**

[outputs/paper_final_verification_20261015/mhw_reference_check.json](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/mhw_reference_check.json:1): reference7670일,1993-01-01–2013-12-31(raw전체10957일의앞round(.7N)). threshold는각DOY±5일의해양SST90thpercentile,clim은동일모집단평균. 고정reference.npz를원자료로재계산하여clim/threshold모두NaN동일성을포함해bitwise일치. 동일SHA `ebf00ed893ac839e1593f81a631c4a51cb2e772840be04c0a40787476c24402e`를모든군에적용한다.

입력용subsetclimatology와별개다. 평가reference에는선택되지않은Train날짜가있지만Validation/Test날짜는없고,이는모든모델에공통인평가label의기준이다. normalization이나forward에이를적용하지않는다. 따라서strictsubset-only**학습통계**조건과모순되지않지만'모든연구단계의정보가subset만'이라는표현은피한다.

코드 [def compute_climatology_threshold(](/ceph_data/yeonwoo/MHW_subset/src/mhw/mhw_skill.py:44), [def day_of_year(](/ceph_data/yeonwoo/MHW_subset/src/mhw/mhw_skill.py:33), [def enforce_min_duration(](/ceph_data/yeonwoo/MHW_subset/src/mhw/mhw_skill.py:100), [def compute_mhw_skill_metrics(](/ceph_data/yeonwoo/MHW_subset/src/mhw/mhw_skill.py:357), [def score(](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96_test/metrics.py:15).

| 검증항목 | 실제 정의/결과 |
|---|---|
| threshold비교 | SST > threshold,동률은비사건 |
| 지속 | 각격자연속5일이상;4일/5일synthetictestPASS |
| gap/smoothing | 짧은gapmerge없음,31일thresholdsmoothing없음 |
| leapday | pandasdayofyear1..366그대로;2019-03-01=60,2020-02-29=60,2020-03-01=61 |
| inputclimatology와차이 | input은2000년month/day정렬;MHW는rawDOY,윤년후1일계절위상차가능 |
| mask | 동일661ocean;coast141,offshore520 |
| TP/FP/FN | 전체고정lead연속targetseries의격자×날짜합산 |
| POD | TP/(TP+FN) |
| FAR | FP/(TP+FP),false-positive-rate(FP/(FP+TN))와다름 |
| CSI | TP/(TP+FP+FN) |
| Precision | TP/(TP+FP) |
| F1 | 2TP/(2TP+FP+FN) |
| 0분모 | 코드정의0;일반적인결측처리와구분 |
| Testdates | lead7 1639일,lead14 1638일,각series1일간격 |

SST는raw°C오차(차이는K),해양격자동일가중RMSE/MAE/Bias이며면적cos(lat)가중평가가아니다. 5일사건은고정lead의여러발행일예측을이어붙인사후판정으로,단일발행시점의미래5일경보능력과동일하지않다. Test경계밖의사건연장일은주어지지않아경계censoring이존재한다. domain-areaevent(.15) 및onsetmatching은grid-day지표와별개다.

rawprediction→재계산숫자,저장dailySSE/TP/FP/FN/labels동일성검사는 [outputs/paper_final_verification_20261015/reanalyze.py](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/reanalyze.py:1)의score및truth비교에있다. 모델·ratio·seed간같은lead의관측label을정확히비교한다. 결과는 [outputs/paper_final_verification_20261015/analysis_status.json](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/analysis_status.json:1), [outputs/paper_final_verification_20261015/test_results120.csv](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/test_results120.csv:1).

이정의는Hobday-style로기술가능하나'표준Hobday전절차(윤년정렬·smoothing·gapmerge)를완전히구현'했다고하면부정확하다. 정의를바꾸는민감도분석은저장예측으로가능하지만이번에는기존평가를조용히대체하지않았다. 현재비교계산오류가확인된것은아니므로재학습사유가아니다.
