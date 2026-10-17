# 재계산·재학습 필요 여부

작성: 2026-10-15 13:59 KST. 신규 학습·재튜닝·원본 코드/데이터/기존 결과 변경 없음.

**현재 Oracle120: PASS WITH LIMITATIONS — 확인된 재학습 대상0개.**

물리계산/전처리/optimizer/학습step/selection에영향을준새오류를확인하지못했다. 120실행프로토콜,53소스해시기록,120best체크포인트SHA,24subsetscale재계산,88stencil테스트가통과했다. 보호654파일해시도일치한다. [outputs/paper_final_verification_20261015/audit_status.json](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/audit_status.json:1), [outputs/paper_final_verification_20261015/protected_integrity.json](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/protected_integrity.json:1), [outputs/paper_final_verification_20261015/rhs_scale_recomputed.csv](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/rhs_scale_recomputed.csv:1).

| 대상 | 판정 | 조치 |
|---|---|---|
| 현재Baseline/P0/NoAdv/Adv/Aux 전체120 | PASS WITH LIMITATIONS | 기존학습유지;이번고정Test및통계보고사용 |
| 기존96Test SST/MHW | PASS | 96/96숫자재현;교체/재학습불필요 |
| Auxiliary24 누락Test | PASS | 이번24개고정추론으로보완완료 |
| 서로다른식의native common_res를동일식처럼비교한주장 | FAIL — RECOMPUTE ONLY (그주장에한함) | 이번공통A/B진단으로대체해서기술;기존수치조용히덮어쓰지않음 |
| WSDM P1–P3 실제실행 및 원고특정감소율 | UNVERIFIED | 해당runmanifest/source/checkpoint와원고표매핑확보전보류 |
| 초기v2/legacy gradient | 과거보정필요경로,현재120과별개 | 현재v3는이미보정;legacy범위는실제실행증거로정해야함 |
| 비균일격자편측미분 일반화 | 현재실험외한계 | 현재균일0.25°재학습사유없음;격자변경시에새검증필요 |
| MHW윤년정렬/정의변경 | 선택적후처리민감도,현재비교오류아님 | 필요시새평가버전으로고정예측재계산;현재기준보존 |
| 명확한제주남서부ROI | UNVERIFIED | 역사적SW-Korea box와혼동금지;결과를보고영역선택하지않음 |

현재'FAIL—RETRAIN REQUIRED'로특정할run은없다. 이것은과거WSDM모든실험의무오류증명이아니다. 예를들어실행시점P2/P3가mask없는Laplacian을사용했거나legacyP0가육지0을미분한것으로확정된다면해당physics학습군의재학습이필요할수있다. 현재정확한P1–P3실행artifact연결이부족하므로추측으로모델·ratio·lead·seed를지정하지않는다. 어떤재학습도시작하지않았다.

최종사용범위:120개모델의SST/MHW성과와공통식잔차진단은한계와함께논문에사용할수있다. 'PINN은언제나저자료량에서우수', '전체정보량동일', 'Adv가완전한실측혼합층수온방정식', '낮은잔차가더정확한예측을보장', 'WSDM P0–P3와동일통제실험'은지지되지않는다.
