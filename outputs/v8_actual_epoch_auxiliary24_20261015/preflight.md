# Preflight PASS

zero-head 최소해는 사용자 지시로 비차단 한계로 재분류. 기존 PASS 보존. CPU24×80epoch sampling/LR 일치 PASS. GPU 대표실행: 독립 초기tensor/원SHA/통계/분할 일치 PASS, 진단 전후 state/RNG/.grad/mode 보존 PASS, 진단ON/OFF 후 실제1step model/optimizer bitwise 일치 PASS, checkpoint roundtrip PASS. 각 run은 동일 초기값/통계/순서를 runtime에서 다시 검사한다. Test 금지.
