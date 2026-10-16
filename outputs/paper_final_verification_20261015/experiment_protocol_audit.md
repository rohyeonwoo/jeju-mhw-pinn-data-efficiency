# Oracle120 학습 프로토콜·전처리·역사적 비교

작성: 2026-10-15 13:59 KST. 신규 학습·재튜닝·원본 코드/데이터/기존 결과 변경 없음.

**Oracle120 120/120 PASS.** 소스해시53기록일치, bestcheckpoint120개SHA일치, 9,600epoch행재검사. [outputs/paper_final_verification_20261015/protocol120.csv](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/protocol120.csv:1), [outputs/paper_final_verification_20261015/audit_status.json](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/audit_status.json:1), [outputs/paper_final_verification_20261015/manifest_checks.json](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/manifest_checks.json:1).

| 항목 | 실제 확인 |
|---|---|
| 조건 | 4ratios×2leads×3seeds×5models=120 |
| 1epoch | 유한DataLoader의선택window정확히1회; drop_last=False |
| Train n | 10%376,25%987,50%1974,100%3995 |
| batch16, 80epochs | 각각24/62/124/250updates/epoch;1920/4960/9920/20000total |
| Validation | 매epoch, data-onlyRMSE, 모두80회 |
| selection | epoch1–80의최저ValRMSE;동점은먼저저장한epoch |
| best range | 4–54, 중앙값22 |
| earlystop | OFF, loop range(1,81);patience81이나actualworker에서는earlystopbreak없음 |
| optimizer | AdamW,betas(.9,.95),wd.1,clipnorm1 |
| LR | initial2e−5,epoch1=5.6e−5,epoch5=2e−4,cosineepoch80=0 |
| physics weight | P0/NoAdv/Adv .1min(epoch/5,1);Aux .0001constant;Baseline0 |
| pairing | Train/Val/Test인덱스·manifest·stats·initialsharedSHA·sampleorder80회·LR80회일치 |

마지막epochLR=0이므로optimizer.step호출은기록되지만그epoch가파라미터학습을추가하는것은아니다. 모든군에공통인기존스케줄이며조기종료가아니다. '80번유효한nonzero-LRepoch'라고기술하지않는다. [src/train_jeju_hilam.py](/ceph_data/yeonwoo/MHW_subset/src/train_jeju_hilam.py:230), [scripts/oracle96_actual/worker.py](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96_actual/worker.py:42), [scripts/oracle96_actual/worker.py](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96_actual/worker.py:69), [scripts/oracle96_actual/worker.py](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96_actual/worker.py:111), [scripts/oracle96_auxiliary24/aux_worker.py](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96_auxiliary24/aux_worker.py:98).

Fullmean/std나FullRHS를덮어쓴흔적없음.120config통계가각manifest와동일하다. v3는v2의input/target통계·climatology·indices를보존하고RHS만보정했다. v2climatology는선택history날짜만,366month/day윤년정렬bin,빈bin은같은subset의주기적선형보간;Fullfallback없음. 10%고유입력608일. input은climatologyfit와같은고유날짜, target통계는선택target날짜. 코드: [src/mhw/subset_climatology.py](/ceph_data/yeonwoo/MHW_subset/src/mhw/subset_climatology.py:19), [src/mhw/subset_climatology.py](/ceph_data/yeonwoo/MHW_subset/src/mhw/subset_climatology.py:26), [src/mhw/subset_climatology.py](/ceph_data/yeonwoo/MHW_subset/src/mhw/subset_climatology.py:95), [src/mhw/subset_preprocessing.py](/ceph_data/yeonwoo/MHW_subset/src/mhw/subset_preprocessing.py:59). 전기간raw를메모리에읽는것과전기간통계fit는구분한다.

선택Traintarget와forcing끝이첫Validationtarget보다이른지모든manifest에서검사. lead7Valtarget2014-01-11–2018-07-06,Test2018-07-07–2022-12-31;lead14Val2014-01-13–2018-07-07,Test2018-07-08–2022-12-31. slidinghistory가분할경계를걸칠수있으므로모든입력관측날짜가분할별완전분리라는주장은하지않는다. [scripts/oracle96/train_mixed_budget_gpu.py](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96/train_mixed_budget_gpu.py:16), [scripts/oracle96/train_mixed_budget_gpu.py](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96/train_mixed_budget_gpu.py:83).

100%는공통7639후보전체가아닌90일완전블록내44일window3995개. 10⊂25⊂50⊂100은6개seed/lead조합모두검사했고Val/Testindices도보존됨.

| 비교항목 | 기존WSDM/Full main 기록 | 해안보정30회/초기v3 | Oracle120 |
|---|---|---|---|
| 구조 | ConvLSTM및SeaCast-inspired모델군 | SeaCast-inspired;baselinehead정책시기별상이 | 모든군동일all-headJejuHiLAM |
| 물리 | P0–P3;Fullmain선정P0 | P0daily/P1daily/P0interval | P0/NoAdv/Adv/Aux별도 |
| seed/lead | main42/7/123,lead1/3/7/14 | 주로lead7 pilot | 42/7/123,lead7/14 |
| train | originalchronological70% | subsetwindows라도Full통계였던pilot;v3는subset통계 | selected3995max,subset-only |
| anomaly | 병합파일사전계산 | Full→subset-only변경 | v2month/day기후값보존 |
| epochs | 기존main저장행12–40등,earlystop서로상이 | 478update가상block,patience12 | 80actual,earlystopOFF |
| λrres | mainPINN10 | pilot주요P0rres0 | Aux만.0001 |
| 검증상태 | P0main일부metric확보;정확한WSDMP1–P3실행mapping미확보 | 보정30회별도checkpoint감사존재 | 120실행해시/config/로그직접확인 |

구형근거: [outputs/main_experiment_final/experiment_manifest_final.json](/ceph_data/yeonwoo/MHW_subset/outputs/main_experiment_final/experiment_manifest_final.json:1), [src/run_main_experiment_final.ps1](/ceph_data/yeonwoo/MHW_subset/src/run_main_experiment_final.ps1:1), [outputs/paper_final_verification_20261015/legacy_inventory.json](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/legacy_inventory.json:1), [outputs/historical_coastal_audit_20261008/report.md](/ceph_data/yeonwoo/MHW_subset/outputs/historical_coastal_audit_20261008/report.md:1). oldFullP0 lead7seed123의S=9.7367229778와보정pilotFullS=.089963214는같은규모가아니다. 전자는해안오염legacy규모와일치하나그숫자만으로WSDMP1–P3실행소스까지확정하지않는다. 현재코드로구형실험을소급인증하지않는다. 기존원고의완전한실행mapping확보전동일통제실험처럼합치면안된다.
