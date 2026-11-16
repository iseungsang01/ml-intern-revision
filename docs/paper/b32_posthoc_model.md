# B.32, read after the fact: the estimators in the first row after a beam blip (not registered)

Frozen B.30 row tables, T_i, cut population, natural cadence, test discharges of the four file-level splits. Rows: the first row after an absent frame with six contiguous rows before it. d = mean departure of the nine BES channels from their 100 ms mean (per-discharge standard deviations). Change = value minus the last visible sample (hold). Intervals: 95 %, 2,000 resamples of discharges.

## D1 (d <= -3)

| split | rows | mean d (sd) | measured change (eV) | predicted: history-only twin | predicted: causal variant | predicted: whole-discharge variant | predicted: causal GP | causal variant vs twin (skill) |
|---|---|---|---|---|---|---|---|---|
| 42 | 85 | -4.41 | -191.0 [-267.8, -133.0] | -3.9 [-19.7, +9.6] | -169.4 [-210.9, -133.8] | -159.2 [-210.7, -113.8] | -12.1 [-30.7, +3.9] | +0.494 [+0.228, +0.708] |
| 1 | 80 | -4.36 | -190.9 [-276.8, -121.7] | -5.4 [-26.7, +13.3] | -163.9 [-193.1, -132.2] | -175.8 [-219.3, -129.3] | -9.6 [-30.3, +10.0] | +0.502 [+0.251, +0.629] |
| 7 | 52 | -4.56 | -243.2 [-336.2, -167.8] | -12.3 [-40.2, +7.7] | -179.5 [-235.7, -126.4] | -142.5 [-215.2, -77.6] | -3.6 [-33.6, +20.5] | +0.677 [+0.580, +0.769] |
| 123 | 83 | -4.42 | -179.9 [-250.9, -120.6] | +4.8 [-2.8, +12.1] | -147.1 [-179.0, -117.5] | -154.1 [-190.9, -116.5] | -3.7 [-18.1, +9.3] | +0.579 [+0.458, +0.657] |

## D2 (-3 < d <= -2)

| split | rows | mean d (sd) | measured change (eV) | predicted: history-only twin | predicted: causal variant | predicted: whole-discharge variant | predicted: causal GP | causal variant vs twin (skill) |
|---|---|---|---|---|---|---|---|---|
| 42 | 35 | -2.47 | -170.7 [-268.6, -86.8] | -9.9 [-56.6, +25.6] | -147.6 [-210.2, -89.3] | -134.5 [-203.5, -75.2] | -8.4 [-58.4, +34.3] | +0.293 [-0.038, +0.498] |
| 1 | 40 | -2.41 | -79.4 [-161.2, +3.3] | +10.2 [-4.9, +26.5] | -68.3 [-97.4, -38.6] | -55.5 [-88.6, -17.5] | +9.8 [-9.4, +35.7] | +0.225 [-0.277, +0.574] |
| 7 | 55 | -2.45 | -231.1 [-292.8, -164.2] | -0.6 [-14.9, +17.8] | -190.6 [-231.6, -138.7] | -145.7 [-190.1, -89.7] | -2.2 [-20.0, +21.2] | +0.566 [+0.438, +0.668] |
| 123 | 38 | -2.51 | -171.7 [-272.3, -69.1] | +4.1 [-19.2, +26.6] | -135.4 [-189.8, -70.5] | -122.2 [-176.7, -54.5] | +3.6 [-33.3, +43.0] | +0.415 [+0.279, +0.577] |

## D3 (d > -2)

| split | rows | mean d (sd) | measured change (eV) | predicted: history-only twin | predicted: causal variant | predicted: whole-discharge variant | predicted: causal GP | causal variant vs twin (skill) |
|---|---|---|---|---|---|---|---|---|
| 42 | 71 | -1.25 | -196.5 [-303.8, -107.5] | -34.0 [-105.3, +6.6] | -160.8 [-235.8, -100.9] | -134.8 [-213.7, -75.0] | -38.9 [-118.9, +13.1] | +0.310 [+0.064, +0.508] |
| 1 | 71 | -1.23 | -126.5 [-234.7, -13.8] | -22.1 [-82.2, +15.2] | -126.0 [-190.0, -72.5] | -96.4 [-151.6, -52.0] | -10.3 [-48.0, +22.3] | +0.162 [+0.011, +0.562] |
| 7 | 76 | -1.22 | -147.0 [-269.3, -48.9] | -53.5 [-121.5, -5.1] | -155.5 [-231.5, -102.0] | -128.9 [-208.6, -72.7] | -14.3 [-99.3, +41.9] | +0.129 [-0.022, +0.309] |
| 123 | 66 | -1.07 | -150.2 [-248.1, -31.1] | -22.7 [-58.5, +5.8] | -168.1 [-226.6, -117.7] | -119.6 [-171.4, -71.9] | -5.0 [-41.9, +27.6] | +0.081 [-0.104, +0.301] |

