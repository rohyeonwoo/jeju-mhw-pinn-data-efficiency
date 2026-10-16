# preprocessing_subset_v3

해안 공간차분 수정,24개 RHS scale 갱신 및 제한 GPU smoke를 통과했습니다. 정식 학습은 실행하지 않았습니다.

[최종 A–E 보고서](../outputs/v3_preflight_20261008/report.md) · [v3 index](../data/subsets/preprocessing_subset_v3/index.json) · [검증 요약](../data/subsets/preprocessing_subset_v3/validation_report.json)

v3는 v2의 climatology NPZ를 경로와 해시로 참조합니다. v2를 함께 보존하세요. Input/Target 통계와 선택 인덱스는 동일하며 RHS만 보정된 해안 stencil로 재계산했습니다. 새 physics 코드에서 legacy v1/v2 RHS를 적용하면 오류로 중단합니다.

Physics only 실행 조건은 `--physics-mode p0_original --lambda-rres 0 --lambda-we-tie 0`으로 명시해야 합니다. 기존72-run v2 실행 계획을 자동 재사용하지 마세요.
