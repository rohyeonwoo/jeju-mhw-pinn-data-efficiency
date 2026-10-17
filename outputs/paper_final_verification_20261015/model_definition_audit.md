# 모델 정의·손실·실행 코드 감사

작성: 2026-10-15 13:59 KST. 신규 학습·재튜닝·원본 코드/데이터/기존 결과 변경 없음.

**판정: 현재 Oracle120 PASS. WSDM P1–P3 실행 시점 연결은 UNVERIFIED.**

Oracle96 96개와 Auxiliary24 24개의 config/history/checkpoint audit를 직접 읽고, Validation-best 파일 SHA를 다시 계산했다. 실행 프로토콜의 소스 해시 기록 53개가 현재 파일과 모두 일치한다(중복 경로를 포함한 기록 수). 프로젝트에는 접근 가능한 Git 저장소가 없으며, 이 검증의 버전 근거는 Git commit이 아니라 저장된 SHA 및 config이다. [outputs/paper_final_verification_20261015/source_hashes.json](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/source_hashes.json:1), [outputs/paper_final_verification_20261015/protocol120.csv](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/protocol120.csv:1). 구형 manifest도 commit을 unknown으로 기록한다.

모두 SeaCast-inspired JejuHiLAM: 1층 GRU(256), 3단계 hierarchical graph, hidden256, processor3, MLP hidden1, direct SST output. 19개 동적 입력과 graph 정적 속성을 사용한다. ConvLSTM과 동일 모델이 아니다. shared 범주(SST head 포함) 14,887,681 + we head66,049 + Rres head66,049 = **15,019,779 parameters**. 전체 `requires_grad` 구조는 같지만 비활성 head는 gradient가 None이다. SST head도 모두 같다.

근거: [src/mhw/models/jeju_hilam.py](/ceph_data/yeonwoo/MHW_subset/src/mhw/models/jeju_hilam.py:114), [src/mhw/models/jeju_hilam.py](/ceph_data/yeonwoo/MHW_subset/src/mhw/models/jeju_hilam.py:167), [src/mhw/models/jeju_hilam.py](/ceph_data/yeonwoo/MHW_subset/src/mhw/models/jeju_hilam.py:219), [src/mhw/models/jeju_hilam.py](/ceph_data/yeonwoo/MHW_subset/src/mhw/models/jeju_hilam.py:326); 실제 공통 초기 tensor/hash 비교는 [def setup(](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96_auxiliary24/aux_worker.py:9) 및 120개 config의 `shared_initial_sha256`, `parameters`, `tensor_fingerprints`.

| 모델 | we head 존재/목적함수 활성 | Rres head 존재/활성 | 실제 목적함수 | 물리 RHS |
|---|---|---|---|---|
| Baseline | 있음/OFF | 있음/OFF | D | 없음 |
| Auxiliary-only | 있음/OFF | 있음/ON | D+0.0001 U | 없음 |
| P0 | 있음/ON | 있음/OFF | D+λ(e)P0 | flux+수평 이류+학습 연행 |
| NoAdv | 있음/OFF | 있음/OFF | D+λ(e)P_A | flux−penetration+MLD 연행 proxy |
| Adv | 있음/OFF | 있음/OFF | D+λ(e)P_B | NoAdv+수평 이류 |

D = mean[(pred_norm−target_norm)²], target_mean/std는 subset 고정값. P = 유효 support 평균[(clip(r,−10,10)/sqrt(S))²]. U = 해양 평균[(86400 Rres)²]/S. S는 `(K/day)²` 단위의 고정 P0 RHS mean-square이며 **std가 아니다**. λ(e)=0.1 min(e/5,1); Aux는 0.0001 고정, 물리 warmup 없음. 모든 모델 entrainment-tie=0, adaptive/adversarial loss 없음.

근거: [def compute_batch_losses(](/ceph_data/yeonwoo/MHW_subset/src/train_jeju_hilam.py:264), [def compute_physics_loss(](/ceph_data/yeonwoo/MHW_subset/src/train_hybrid_jeju_forecast.py:447), [def rres_penalty(](/ceph_data/yeonwoo/MHW_subset/src/train_hybrid_jeju_forecast.py:1081), [def arguments(](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96_actual/worker.py:13), [def setup(](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96_auxiliary24/aux_worker.py:9).

Gradient: D→SST head→shared. P0→SST interpolation/derivative 및 we head→shared. NoAdv/Adv→예측 SST interpolation→shared; 두 auxiliary head는 사용하지 않는다. U→Rres head→shared이나 SST output head에는 U의 직접 경로가 없다. 초기 및 각 학습 epoch의 physics predictor gradient 기록이 양수인 검사와 Aux autograd 진단을 보존했다. `Physics only`와 이번 `Adv`는 head 수의 차이가 아닌 RHS의 차이다. 과거 `Full PINN`은 physics+Rres이고 **이번 Adv는 Full PINN(aux ON)이 아니다**.

NoAdv/Adv는 동일 모듈 `path(..., advection=False/True)`의 단일 분기로만 다르다. NoAdv에서도 항 진단을 위해 gradient/advection 숫자는 계산하지만 loss RHS에는 포함하지 않아 그 항의 gradient는 전달되지 않는다. 공통 mask와 RHS scale도 같다. [scripts/oracle96/mixed_budget_physics.py](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96/mixed_budget_physics.py:25), [scripts/oracle96/mixed_budget_physics.py](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96/mixed_budget_physics.py:30), [scripts/oracle96/mixed_budget_physics.py](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96/mixed_budget_physics.py:42).

P0와 Adv의 차이: 단파 투과의 유무, 학습 we×1K와 관측 MLD-deepening×(NOAA proxy−Tb), 기존 budget2d 산술 ML 유속과 3D 두께가중 ML 유속. 따라서 P0→Adv 결과를 '이류 추가 효과'로 해석하면 안 된다.

구형 WSDM 정의 매핑은 P0=original, P1=original+vertical, P2=original+diffusion, P3=original+vertical+diffusion. [src/train_hybrid_jeju_forecast.py](/ceph_data/yeonwoo/MHW_subset/src/train_hybrid_jeju_forecast.py:55). 현재 NoAdv/Adv에 P1–P3의 vertical/diffusion은 없다. 구형 원고·P1–P3 전용 실행 config/checkpoint/source-hash 연결을 확보하지 못했으므로 동일 실험이라는 주장은 **UNVERIFIED**. [outputs/paper_final_verification_20261015/legacy_inventory.json](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/legacy_inventory.json:1) 및 experiment_protocol_audit.md 참고.
