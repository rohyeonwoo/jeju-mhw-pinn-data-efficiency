# Actual Epoch Oracle96 protocol

80 actual epochs; one shuffled no-replacement pass per epoch; batch16/drop_last=False. Early stopping OFF. LR initial2e-5, peak2e-4, five actual-epoch linear warmup then original cosine to epoch80 LR=0. Physics lambda is the original Oracle96 config (Baseline0, P0/NoAdv/Adv0.1) times min(epoch/5,1). NoAdv/Adv refer to absence/presence of horizontal advection, not adversarial losses. No auxiliary/adversarial schedule is added.

Official checkpoint: minimum Validation RMSE across epochs1–80. Diagnostic post-warmup best: epoch>=5 (epoch5 already uses full lambda on every update). Validation every actual epoch, no physics or forcing in selection; extra validation calls at completion only verify checkpoints and forcing independence. Test execution disabled.

Same initialization, subset, statistics, predictor, physics terms, optimizer AdamW betas(.9,.95), weight_decay.1 and clipnorm1 as frozen Oracle96. 96 runs initialized from seed; no old checkpoint reuse. Each GPU7/8/9 max two workers, 14GiB allocator budget each plus4GiB headroom; pause new launches if insufficient memory. Old/new locks prevent overlap.

Compared with old: actual epochs replace478-update cycles; physics warmup old0 -> new5 actual epochs; LR formula was epoch-based, not a per-update scheduler, and is now called once per actual epoch. Epoch80 LR0 remains intentional per unchanged formula; total updates count optimizer calls. Training RMSE is online batch aggregation.

SST-label efficiency with training-only oracle, not equal total-information efficiency. Same prior Test has informed earlier research decisions; future evaluation is exploratory. Test will require a separate plan/authorization after96 completion.
