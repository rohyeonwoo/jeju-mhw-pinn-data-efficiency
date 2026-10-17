# 제주 연안 PINN 최종 정합성 검증

작성: 2026-10-15 13:59 KST. 신규 학습·재튜닝·원본 코드/데이터/기존 결과 변경 없음.

**현재 Oracle120은 PASS WITH LIMITATIONS입니다. 확인된 재학습 필요 모델은0개입니다. 기존 WSDM P1–P3의 실행 시점 정합성은 별도로 UNVERIFIED입니다.**

이번 작업은 신규 학습/튜닝 없이 수행했습니다. CPU 수치·로그·예측 검산과 저장된 Auxiliary-only24개 모델의 Test 추론만 실행했습니다. GPU5/6/7에 각1개씩 사용했고 모든GPU PID의 process name `yeonwoo`를 확인했습니다. 원본 코드·데이터·통계·체크포인트·기존 결과를 덮어쓰지 않았으며 보호 대상654개 SHA가 유지됐습니다.

주요 검증 결과:

- **88/88 이류·해안 수치 테스트 PASS.** 실제격자는0.25°균일, 미터·cos(latitude)·K/day변환이일관됩니다. 육지값 변경 영향0, 과거보정stencil과동일. 위도속성의`decreasing`문구는실제증가좌표와불일치하지만 계산은실제좌표값을사용합니다.
- **120/120 학습 프로토콜 PASS.** 80actualepoch,동일subset/통계/초기Predictor/shuffle/LR,Validation-best확인. 실행소스해시53기록및bestcheckpoint120개해시일치.
- **24/24 RHS scale 재계산 PASS.** subset P0 RHS mean-square이며모든arm이같은sqrt(S)로정규화합니다. 'std'나'모델별고유RHSscale'로기술하면안됩니다.
- **120/120 Test SST·MHW 재검산 PASS.** 고정reference는1993–2013이며threshold/clim을원자료로재생성해비트단위일치. 모델/ratio/seed간동일truth.
- **Auxiliary Test24개 완료:** Baseline보다RMSE가낮은조건15/24;Validation은16/24.25%·Lead14는평균약2.47%개선,3/3seed. 전체적으로큰효과가일관되게나타나지는않습니다.
- **NoAdv 대 Adv:** RMSE22/24,CSI19/24에서NoAdv우위.8개ratio/lead평균RMSE모두NoAdv가낮지만90일시간CI에서0을제외하는것은6/8입니다. Adv의공통이류포함잔차가더작다는결과와예측우위는별개입니다.

| 검증 항목 | 판정 | 근거 | 논문에 미치는 영향 |
| --- | --- | --- | --- |
| 수평 이류항 수식 | PASS | −86400(uTx+vTy), 선형장/속도반전 통과 | 현재 Adv 이류 구현 사용 가능 |
| 단위 및 부호 | PASS WITH LIMITATIONS | 미터·cos(lat)·86400 확인; upstream 원자료 처리 일부 미확인 | K/day 서술 가능; 저장 방향 attrs 오류와 원자료 한계 명시 |
| 해안 경계 계산 | PASS WITH LIMITATIONS | 88/88; 661 ocean/660 support; 비균일 편측 한계 | 실제 균일 격자에서 유효; 일반 불균일 격자로 확대 금지 |
| 물리 RHS Scale | PASS | 24/24 재계산 일치; subset P0 mean-square | 공통 RMS normalization으로 정확히 기술 |
| 모델별 손실 구성 | PASS | 전체 구조 동일; Aux만 Rres; NoAdv/Adv 이류만 차이 | Adv를 physics+aux Full PINN으로 부르지 않음 |
| 120개 학습 프로토콜 | PASS | 120/120; 9600epoch; 동일 pairing; best SHA | 100%=3995, actual epoch, 마지막 LR0 명시 |
| SST 평가 | PASS | 120개 raw 예측 재검산 및 dates/targets 일치 | 고정 Validation-best의 탐색적 Test 비교 |
| MHW 평가 | PASS WITH LIMITATIONS | 동일고정threshold 재생성일치; rawDOY·5일 | 윤년/경계/고정lead사후사건 정의 한계 |
| Auxiliary 기여 | PASS WITH LIMITATIONS | Val16/24,Test15/24; gradient 점차약화 | 보편적 큰효과 주장불가;25%·L14조건 개선 |
| NoAdv–Adv 비교 | PASS WITH LIMITATIONS | RMSE22/24,CSI19/24;시간blockCI | 해당조건에서 이류제외 예측우위;물리법칙 부정아님 |
| 기존 WSDM과의 정합성 | UNVERIFIED | P0–P3 매핑있으나 정확한실행source/checkpoint미확보 | 현재120과별도실험군;기존물리감소율확정금지 |

**모델별 Test RMSE(K),3seed평균±SD**

| Ratio | Lead | Baseline | Auxiliary | P0 | NoAdv | Adv |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 7 | 1.00989 ± 0.10863 | 1.02272 ± 0.09895 | 1.04071 ± 0.06600 | 0.97181 ± 0.04737 | 1.03338 ± 0.03076 |
| 10 | 14 | 1.27636 ± 0.07437 | 1.27187 ± 0.03861 | 1.24328 ± 0.03480 | 1.26093 ± 0.03555 | 1.29221 ± 0.05898 |
| 25 | 7 | 0.86134 ± 0.01735 | 0.86054 ± 0.01786 | 0.87512 ± 0.03521 | 0.85559 ± 0.01025 | 0.89267 ± 0.03360 |
| 25 | 14 | 1.12747 ± 0.03039 | 1.09920 ± 0.01371 | 1.13814 ± 0.05339 | 1.07296 ± 0.01705 | 1.12340 ± 0.03892 |
| 50 | 7 | 0.84315 ± 0.01643 | 0.84295 ± 0.01576 | 0.84852 ± 0.01102 | 0.83861 ± 0.01219 | 0.88028 ± 0.02951 |
| 50 | 14 | 1.01899 ± 0.01535 | 1.01762 ± 0.01613 | 1.06239 ± 0.02349 | 1.05851 ± 0.00602 | 1.07103 ± 0.02858 |
| 100 | 7 | 0.79605 ± 0.00953 | 0.79584 ± 0.00939 | 0.83936 ± 0.00907 | 0.82853 ± 0.01925 | 0.85151 ± 0.00686 |
| 100 | 14 | 0.99361 ± 0.01131 | 0.99341 ± 0.01097 | 1.00220 ± 0.01482 | 0.99069 ± 0.00313 | 1.00709 ± 0.00349 |

논문에 사용할 수 있는 해석: **동일한 SST label과학습프로토콜에서,이자료·해상도·대리수온·가중치조건의NoAdv는Adv보다대체로좋은예측성능을보였습니다. 이는이류물리법칙이틀렸다는검증이아닙니다.** PINN은Training-only oracleforcing을추가감독으로쓰므로SST-label efficiency실험이며total-information efficiency가아닙니다. Auxiliary의감소하는gradient와작은효과도ablation결과로그대로보고해야합니다.

필수 한계: NOAA SST≠GLORYSML평균,전체순단파단일지수근사와Kd=.06의미보정성,CPU투과on/off잔차부호반전에따른가열과보정가능성,원격자처리provenance부족,rawDOY MHW윤년차이,고정lead예측series의사후사건판정,고정3seed조건부CI와다중비교/Test반복열람. 정확한'제주남서부'ROI는확정하지못했으며기존SW-Korea box분석과구분합니다.

**산출물**

- 모델/물리: [outputs/paper_final_verification_20261015/model_definition_audit.md](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/model_definition_audit.md:1), [outputs/paper_final_verification_20261015/physics_equation_audit.md](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/physics_equation_audit.md:1)
- 이류/해안: [outputs/paper_final_verification_20261015/horizontal_advection_audit.md](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/horizontal_advection_audit.md:1), [outputs/paper_final_verification_20261015/coastal_stencil_tests.md](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/coastal_stencil_tests.md:1)
- 프로토콜/평가: [outputs/paper_final_verification_20261015/experiment_protocol_audit.md](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/experiment_protocol_audit.md:1), [outputs/paper_final_verification_20261015/mhw_evaluation_audit.md](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/mhw_evaluation_audit.md:1)
- Auxiliary/통계: [outputs/paper_final_verification_20261015/auxiliary_ablation_audit.md](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_ablation_audit.md:1), [outputs/paper_final_verification_20261015/statistical_comparison.md](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/statistical_comparison.md:1)
- 최종조치: [outputs/paper_final_verification_20261015/retraining_decision.md](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/retraining_decision.md:1)
- 전체120수치: [outputs/paper_final_verification_20261015/test_results120.csv](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/test_results120.csv:1), [outputs/paper_final_verification_20261015/comparison_summary.csv](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/comparison_summary.csv:1)
- 전체대응비교: [outputs/paper_final_verification_20261015/paired_seed_results.csv](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/paired_seed_results.csv:1), [outputs/paper_final_verification_20261015/block_bootstrap.csv](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/block_bootstrap.csv:1)

과거모델별native residual을같은공통식의잔차로비교한주장은그대로사용할수없습니다. 이번보고서는모든120개예측에공통A/B식각각을적용한새진단을따로제공합니다. 특정WSDM원고표의감소율은원고/runmapping확보전확정하지않습니다.
