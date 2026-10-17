# Main Experiment Final Summary

Final physics for PINN models: P0.

## SST Headline

| model         | lead_days | n_runs | test_rmse_mean | test_rmse_std | test_acc_mean | test_acc_std | mhw_grid_csi_mean | mhw_grid_csi_std | tail_strong_plus_csi_mean | tail_strong_plus_csi_std |
| ------------- | --------- | ------ | -------------- | ------------- | ------------- | ------------ | ----------------- | ---------------- | ------------------------- | ------------------------ |
| convlstm      | 1         | 3      | 0.6076         | 0.0361        | 0.8677        | 0.0157       | 0.4873            | 0.0326           | 0.3014                    | 0.0361                   |
| convlstm_pinn | 1         | 3      | 0.5096         | 0.0403        | 0.9025        | 0.0185       | 0.6137            | 0.0265           | 0.4236                    | 0.0245                   |
| jeju_hilam    | 1         | 3      | 0.2910         | 0.0034        | 0.9676        | 0.0008       | 0.8014            | 0.0096           | 0.6124                    | 0.0206                   |
| persistence   | 1         | 1      | 0.3191         | 0.0000        | 0.9614        | 0.0000       | 0.8250            | 0.0000           | 0.6360                    | 0.0000                   |
| pi_jeju_hilam | 1         | 3      | 0.3127         | 0.0062        | 0.9624        | 0.0015       | 0.7656            | 0.0161           | 0.5729                    | 0.0193                   |
| convlstm      | 3         | 3      | 0.7734         | 0.0425        | 0.7582        | 0.0283       | 0.4311            | 0.0172           | 0.2105                    | 0.0242                   |
| convlstm_pinn | 3         | 3      | 0.7758         | 0.0556        | 0.7566        | 0.0192       | 0.3546            | 0.0102           | 0.1577                    | 0.0123                   |
| jeju_hilam    | 3         | 3      | 0.5651         | 0.0020        | 0.8707        | 0.0010       | 0.5576            | 0.0144           | 0.2566                    | 0.0178                   |
| persistence   | 3         | 1      | 0.6507         | 0.0000        | 0.8427        | 0.0000       | 0.6039            | 0.0000           | 0.3554                    | 0.0000                   |
| pi_jeju_hilam | 3         | 3      | 0.5676         | 0.0060        | 0.8696        | 0.0030       | 0.5624            | 0.0155           | 0.2547                    | 0.0182                   |
| convlstm      | 7         | 3      | 0.9687         | 0.0350        | 0.6283        | 0.0404       | 0.3265            | 0.0433           | 0.1256                    | 0.0230                   |
| convlstm_pinn | 7         | 3      | 1.0272         | 0.1080        | 0.6221        | 0.0140       | 0.3240            | 0.0058           | 0.1250                    | 0.0006                   |
| jeju_hilam    | 7         | 3      | 0.7854         | 0.0088        | 0.7316        | 0.0087       | 0.3642            | 0.0321           | 0.1142                    | 0.0268                   |
| persistence   | 7         | 1      | 1.0332         | 0.0000        | 0.6456        | 0.0000       | 0.3830            | 0.0000           | 0.1696                    | 0.0000                   |
| pi_jeju_hilam | 7         | 3      | 0.7920         | 0.0036        | 0.7252        | 0.0026       | 0.3386            | 0.0127           | 0.0864                    | 0.0091                   |
| convlstm      | 14        | 3      | 1.1407         | 0.0485        | 0.4690        | 0.0088       | 0.2378            | 0.0160           | 0.0711                    | 0.0060                   |
| convlstm_pinn | 14        | 3      | 1.2235         | 0.0539        | 0.4743        | 0.0239       | 0.2439            | 0.0047           | 0.0848                    | 0.0072                   |
| jeju_hilam    | 14        | 3      | 0.9454         | 0.0151        | 0.5859        | 0.0237       | 0.2492            | 0.0474           | 0.0628                    | 0.0267                   |
| persistence   | 14        | 1      | 1.5999         | 0.0000        | 0.3984        | 0.0000       | 0.2538            | 0.0000           | 0.0700                    | 0.0000                   |
| pi_jeju_hilam | 14        | 3      | 0.9632         | 0.0063        | 0.5614        | 0.0118       | 0.2242            | 0.0344           | 0.0507                    | 0.0215                   |

## Paired Deltas

| comparison                      | lead_days | metric       | n_pairs | delta_mean | delta_std | improved_pairs |
| ------------------------------- | --------- | ------------ | ------- | ---------- | --------- | -------------- |
| ConvLSTM - Persistence          | 1         | mhw_grid_csi | 3       | -0.3377    | 0.0326    | 0              |
| ConvLSTM - Persistence          | 1         | mhw_grid_far | 3       | 0.2145     | 0.0197    | 0              |
| ConvLSTM - Persistence          | 1         | test_acc     | 3       | -0.0938    | 0.0157    | 0              |
| ConvLSTM - Persistence          | 1         | test_rmse    | 3       | 0.2885     | 0.0361    | 0              |
| ConvLSTM-PINN(P0) - ConvLSTM    | 1         | mhw_grid_csi | 3       | 0.1264     | 0.0219    | 3              |
| ConvLSTM-PINN(P0) - ConvLSTM    | 1         | mhw_grid_far | 3       | -0.1614    | 0.0488    | 3              |
| ConvLSTM-PINN(P0) - ConvLSTM    | 1         | test_acc     | 3       | 0.0349     | 0.0256    | 3              |
| ConvLSTM-PINN(P0) - ConvLSTM    | 1         | test_rmse    | 3       | -0.0980    | 0.0429    | 3              |
| ConvLSTM-PINN(P0) - Persistence | 1         | mhw_grid_csi | 3       | -0.2113    | 0.0265    | 0              |
| ConvLSTM-PINN(P0) - Persistence | 1         | mhw_grid_far | 3       | 0.0530     | 0.0293    | 0              |
| ConvLSTM-PINN(P0) - Persistence | 1         | test_acc     | 3       | -0.0589    | 0.0185    | 0              |
| ConvLSTM-PINN(P0) - Persistence | 1         | test_rmse    | 3       | 0.1905     | 0.0403    | 0              |
| JejuHiLAM - Persistence         | 1         | mhw_grid_csi | 3       | -0.0236    | 0.0096    | 0              |
| JejuHiLAM - Persistence         | 1         | mhw_grid_far | 3       | -0.0378    | 0.0125    | 3              |
| JejuHiLAM - Persistence         | 1         | test_acc     | 3       | 0.0061     | 0.0008    | 3              |
| JejuHiLAM - Persistence         | 1         | test_rmse    | 3       | -0.0281    | 0.0034    | 3              |
| PI-JejuHiLAM(P0) - JejuHiLAM    | 1         | mhw_grid_csi | 3       | -0.0357    | 0.0182    | 0              |
| PI-JejuHiLAM(P0) - JejuHiLAM    | 1         | mhw_grid_far | 3       | 0.0058     | 0.0072    | 1              |
| PI-JejuHiLAM(P0) - JejuHiLAM    | 1         | test_acc     | 3       | -0.0052    | 0.0023    | 0              |
| PI-JejuHiLAM(P0) - JejuHiLAM    | 1         | test_rmse    | 3       | 0.0217     | 0.0095    | 0              |
| PI-JejuHiLAM(P0) - Persistence  | 1         | mhw_grid_csi | 3       | -0.0593    | 0.0161    | 0              |
| PI-JejuHiLAM(P0) - Persistence  | 1         | mhw_grid_far | 3       | -0.0320    | 0.0081    | 3              |
| PI-JejuHiLAM(P0) - Persistence  | 1         | test_acc     | 3       | 0.0009     | 0.0015    | 2              |
| PI-JejuHiLAM(P0) - Persistence  | 1         | test_rmse    | 3       | -0.0064    | 0.0062    | 3              |
| ConvLSTM - Persistence          | 3         | mhw_grid_csi | 3       | -0.1727    | 0.0172    | 0              |
| ConvLSTM - Persistence          | 3         | mhw_grid_far | 3       | 0.0517     | 0.0507    | 0              |
| ConvLSTM - Persistence          | 3         | test_acc     | 3       | -0.0845    | 0.0283    | 0              |
| ConvLSTM - Persistence          | 3         | test_rmse    | 3       | 0.1227     | 0.0425    | 0              |
| ConvLSTM-PINN(P0) - ConvLSTM    | 3         | mhw_grid_csi | 3       | -0.0765    | 0.0145    | 0              |
| ConvLSTM-PINN(P0) - ConvLSTM    | 3         | mhw_grid_far | 3       | -0.0172    | 0.1454    | 2              |
| ConvLSTM-PINN(P0) - ConvLSTM    | 3         | test_acc     | 3       | -0.0016    | 0.0163    | 1              |
| ConvLSTM-PINN(P0) - ConvLSTM    | 3         | test_rmse    | 3       | 0.0025     | 0.0493    | 1              |
| ConvLSTM-PINN(P0) - Persistence | 3         | mhw_grid_csi | 3       | -0.2493    | 0.0102    | 0              |
| ConvLSTM-PINN(P0) - Persistence | 3         | mhw_grid_far | 3       | 0.0345     | 0.1120    | 2              |
| ConvLSTM-PINN(P0) - Persistence | 3         | test_acc     | 3       | -0.0861    | 0.0192    | 0              |
| ConvLSTM-PINN(P0) - Persistence | 3         | test_rmse    | 3       | 0.1251     | 0.0556    | 0              |
| JejuHiLAM - Persistence         | 3         | mhw_grid_csi | 3       | -0.0463    | 0.0144    | 0              |
| JejuHiLAM - Persistence         | 3         | mhw_grid_far | 3       | -0.1320    | 0.0259    | 3              |
| JejuHiLAM - Persistence         | 3         | test_acc     | 3       | 0.0280     | 0.0010    | 3              |
| JejuHiLAM - Persistence         | 3         | test_rmse    | 3       | -0.0856    | 0.0020    | 3              |
| PI-JejuHiLAM(P0) - JejuHiLAM    | 3         | mhw_grid_csi | 3       | 0.0048     | 0.0280    | 1              |
| PI-JejuHiLAM(P0) - JejuHiLAM    | 3         | mhw_grid_far | 3       | 0.0136     | 0.0243    | 1              |
| PI-JejuHiLAM(P0) - JejuHiLAM    | 3         | test_acc     | 3       | -0.0011    | 0.0020    | 1              |
| PI-JejuHiLAM(P0) - JejuHiLAM    | 3         | test_rmse    | 3       | 0.0025     | 0.0040    | 1              |
| PI-JejuHiLAM(P0) - Persistence  | 3         | mhw_grid_csi | 3       | -0.0414    | 0.0155    | 0              |
| PI-JejuHiLAM(P0) - Persistence  | 3         | mhw_grid_far | 3       | -0.1184    | 0.0175    | 3              |
| PI-JejuHiLAM(P0) - Persistence  | 3         | test_acc     | 3       | 0.0269     | 0.0030    | 3              |
| PI-JejuHiLAM(P0) - Persistence  | 3         | test_rmse    | 3       | -0.0831    | 0.0060    | 3              |
| ConvLSTM - Persistence          | 7         | mhw_grid_csi | 3       | -0.0565    | 0.0433    | 0              |
| ConvLSTM - Persistence          | 7         | mhw_grid_far | 3       | -0.0309    | 0.0279    | 3              |
| ConvLSTM - Persistence          | 7         | test_acc     | 3       | -0.0173    | 0.0404    | 1              |
| ConvLSTM - Persistence          | 7         | test_rmse    | 3       | -0.0645    | 0.0350    | 3              |
| ConvLSTM-PINN(P0) - ConvLSTM    | 7         | mhw_grid_csi | 3       | -0.0025    | 0.0375    | 1              |
| ConvLSTM-PINN(P0) - ConvLSTM    | 7         | mhw_grid_far | 3       | 0.0378     | 0.0714    | 1              |
| ConvLSTM-PINN(P0) - ConvLSTM    | 7         | test_acc     | 3       | -0.0062    | 0.0296    | 1              |
| ConvLSTM-PINN(P0) - ConvLSTM    | 7         | test_rmse    | 3       | 0.0585     | 0.1108    | 1              |
| ConvLSTM-PINN(P0) - Persistence | 7         | mhw_grid_csi | 3       | -0.0590    | 0.0058    | 0              |
| ConvLSTM-PINN(P0) - Persistence | 7         | mhw_grid_far | 3       | 0.0069     | 0.0802    | 1              |
| ConvLSTM-PINN(P0) - Persistence | 7         | test_acc     | 3       | -0.0235    | 0.0140    | 0              |
| ConvLSTM-PINN(P0) - Persistence | 7         | test_rmse    | 3       | -0.0060    | 0.1080    | 2              |
| JejuHiLAM - Persistence         | 7         | mhw_grid_csi | 3       | -0.0188    | 0.0321    | 1              |
| JejuHiLAM - Persistence         | 7         | mhw_grid_far | 3       | -0.1932    | 0.0317    | 3              |
| JejuHiLAM - Persistence         | 7         | test_acc     | 3       | 0.0860     | 0.0087    | 3              |
| JejuHiLAM - Persistence         | 7         | test_rmse    | 3       | -0.2479    | 0.0088    | 3              |
| PI-JejuHiLAM(P0) - JejuHiLAM    | 7         | mhw_grid_csi | 3       | -0.0255    | 0.0271    | 0              |
| PI-JejuHiLAM(P0) - JejuHiLAM    | 7         | mhw_grid_far | 3       | -0.0236    | 0.0254    | 3              |
| PI-JejuHiLAM(P0) - JejuHiLAM    | 7         | test_acc     | 3       | -0.0064    | 0.0097    | 1              |
| PI-JejuHiLAM(P0) - JejuHiLAM    | 7         | test_rmse    | 3       | 0.0067     | 0.0106    | 1              |
| PI-JejuHiLAM(P0) - Persistence  | 7         | mhw_grid_csi | 3       | -0.0444    | 0.0127    | 0              |
| PI-JejuHiLAM(P0) - Persistence  | 7         | mhw_grid_far | 3       | -0.2168    | 0.0205    | 3              |
| PI-JejuHiLAM(P0) - Persistence  | 7         | test_acc     | 3       | 0.0796     | 0.0026    | 3              |
| PI-JejuHiLAM(P0) - Persistence  | 7         | test_rmse    | 3       | -0.2412    | 0.0036    | 3              |
| ConvLSTM - Persistence          | 14        | mhw_grid_csi | 3       | -0.0160    | 0.0160    | 0              |
| ConvLSTM - Persistence          | 14        | mhw_grid_far | 3       | -0.0965    | 0.0167    | 3              |
| ConvLSTM - Persistence          | 14        | test_acc     | 3       | 0.0707     | 0.0088    | 3              |
| ConvLSTM - Persistence          | 14        | test_rmse    | 3       | -0.4592    | 0.0485    | 3              |
| ConvLSTM-PINN(P0) - ConvLSTM    | 14        | mhw_grid_csi | 3       | 0.0061     | 0.0132    | 1              |
| ConvLSTM-PINN(P0) - ConvLSTM    | 14        | mhw_grid_far | 3       | 0.0561     | 0.0296    | 0              |
| ConvLSTM-PINN(P0) - ConvLSTM    | 14        | test_acc     | 3       | 0.0053     | 0.0152    | 1              |
| ConvLSTM-PINN(P0) - ConvLSTM    | 14        | test_rmse    | 3       | 0.0828     | 0.0954    | 1              |
| ConvLSTM-PINN(P0) - Persistence | 14        | mhw_grid_csi | 3       | -0.0099    | 0.0047    | 0              |
| ConvLSTM-PINN(P0) - Persistence | 14        | mhw_grid_far | 3       | -0.0404    | 0.0330    | 3              |
| ConvLSTM-PINN(P0) - Persistence | 14        | test_acc     | 3       | 0.0759     | 0.0239    | 3              |
| ConvLSTM-PINN(P0) - Persistence | 14        | test_rmse    | 3       | -0.3764    | 0.0539    | 3              |
| JejuHiLAM - Persistence         | 14        | mhw_grid_csi | 3       | -0.0046    | 0.0474    | 2              |
| JejuHiLAM - Persistence         | 14        | mhw_grid_far | 3       | -0.2253    | 0.0117    | 3              |
| JejuHiLAM - Persistence         | 14        | test_acc     | 3       | 0.1875     | 0.0237    | 3              |
| JejuHiLAM - Persistence         | 14        | test_rmse    | 3       | -0.6545    | 0.0151    | 3              |
| PI-JejuHiLAM(P0) - JejuHiLAM    | 14        | mhw_grid_csi | 3       | -0.0251    | 0.0272    | 0              |
| PI-JejuHiLAM(P0) - JejuHiLAM    | 14        | mhw_grid_far | 3       | -0.0149    | 0.0315    | 2              |
| PI-JejuHiLAM(P0) - JejuHiLAM    | 14        | test_acc     | 3       | -0.0244    | 0.0240    | 0              |
| PI-JejuHiLAM(P0) - JejuHiLAM    | 14        | test_rmse    | 3       | 0.0177     | 0.0136    | 0              |
| PI-JejuHiLAM(P0) - Persistence  | 14        | mhw_grid_csi | 3       | -0.0297    | 0.0344    | 0              |
| PI-JejuHiLAM(P0) - Persistence  | 14        | mhw_grid_far | 3       | -0.2402    | 0.0211    | 3              |
| PI-JejuHiLAM(P0) - Persistence  | 14        | test_acc     | 3       | 0.1631     | 0.0118    | 3              |
| PI-JejuHiLAM(P0) - Persistence  | 14        | test_rmse    | 3       | -0.6367    | 0.0063    | 3              |
