# B.30 row tables: sensitivity to suspicious rows (post hoc, not registered)

T_i, skill a vs b = 1 - SSE_a / SSE_b. Each cell: range of the four file-level splits and the number of splits whose 95 % interval (2,000 discharge-level resamples) excludes 0. A row is left out when the stated condition fails for the scored value or for the last visible sample.

## cut, a CES sample visible every 10 ms

| rows kept | share of rows | bb80 vs hist80 | causal vs hist80 | bb vs gp | causal vs gp | bb vs hold |
|---|---|---|---|---|---|---|
| all | 1.000 to 1.000 | +0.064 to +0.113 (4/4) | +0.027 to +0.115 (4/4) | +0.077 to +0.134 (4/4) | +0.068 to +0.126 (4/4) | +0.358 to +0.411 (4/4) |
| all four exclusions | 0.885 to 0.898 | +0.084 to +0.145 (4/4) | +0.044 to +0.152 (4/4) | +0.084 to +0.136 (4/4) | +0.069 to +0.126 (4/4) | +0.294 to +0.354 (4/4) |
| all four exclusions and no row within 100 ms after a blip | 0.836 to 0.846 | +0.080 to +0.140 (3/4) | +0.043 to +0.148 (4/4) | +0.076 to +0.130 (3/4) | +0.064 to +0.112 (4/4) | +0.302 to +0.360 (4/4) |
| jump <= 1 keV | 0.985 to 0.993 | +0.065 to +0.116 (3/4) | +0.053 to +0.120 (3/4) | +0.050 to +0.124 (3/4) | +0.019 to +0.128 (3/4) | +0.236 to +0.298 (4/4) |
| not within 100 ms after a beam blip (absent frame) | 0.940 to 0.945 | +0.058 to +0.105 (4/4) | +0.026 to +0.107 (4/4) | +0.066 to +0.127 (4/4) | +0.055 to +0.106 (4/4) | +0.358 to +0.414 (4/4) |
| not within 30 ms after a low-BES row | 0.977 to 0.983 | +0.065 to +0.122 (4/4) | +0.026 to +0.124 (4/4) | +0.103 to +0.141 (4/4) | +0.099 to +0.137 (4/4) | +0.380 to +0.422 (4/4) |
| only within 100 ms after a beam blip (absent frame) | 0.055 to 0.060 | +0.086 to +0.184 (4/4) | +0.040 to +0.187 (2/4) | +0.182 to +0.288 (4/4) | +0.131 to +0.291 (3/4) | +0.360 to +0.435 (4/4) |
| values <= 2.5 keV | 0.995 to 0.998 | +0.074 to +0.133 (4/4) | +0.036 to +0.133 (4/4) | +0.091 to +0.138 (4/4) | +0.084 to +0.138 (4/4) | +0.364 to +0.410 (4/4) |
| values >= 150 eV | 0.909 to 0.915 | +0.063 to +0.109 (4/4) | +0.019 to +0.105 (3/4) | +0.080 to +0.121 (4/4) | +0.069 to +0.113 (4/4) | +0.348 to +0.402 (4/4) |

## cut, a CES sample visible every 50 ms

| rows kept | share of rows | bb80 vs hist80 | causal vs hist80 | bb vs gp | causal vs gp | bb vs hold |
|---|---|---|---|---|---|---|
| all | 1.000 to 1.000 | +0.076 to +0.156 (4/4) | +0.054 to +0.156 (4/4) | +0.108 to +0.129 (4/4) | +0.070 to +0.134 (4/4) | +0.298 to +0.345 (4/4) |
| all four exclusions | 0.886 to 0.900 | +0.084 to +0.156 (3/4) | +0.084 to +0.147 (4/4) | +0.038 to +0.087 (3/4) | +0.019 to +0.088 (3/4) | +0.170 to +0.254 (4/4) |
| all four exclusions and no row within 100 ms after a blip | 0.836 to 0.847 | +0.069 to +0.151 (3/4) | +0.080 to +0.141 (4/4) | +0.029 to +0.082 (3/4) | +0.016 to +0.086 (3/4) | +0.165 to +0.251 (4/4) |
| jump <= 1 keV | 0.983 to 0.992 | +0.079 to +0.135 (4/4) | +0.078 to +0.133 (4/4) | +0.015 to +0.070 (3/4) | -0.001 to +0.063 (1/4) | +0.126 to +0.224 (4/4) |
| not within 100 ms after a beam blip (absent frame) | 0.939 to 0.944 | +0.061 to +0.151 (4/4) | +0.047 to +0.149 (4/4) | +0.101 to +0.124 (4/4) | +0.069 to +0.125 (4/4) | +0.289 to +0.343 (4/4) |
| not within 30 ms after a low-BES row | 0.979 to 0.985 | +0.078 to +0.161 (4/4) | +0.059 to +0.160 (4/4) | +0.109 to +0.135 (4/4) | +0.074 to +0.145 (4/4) | +0.307 to +0.349 (4/4) |
| only within 100 ms after a beam blip (absent frame) | 0.056 to 0.061 | +0.144 to +0.252 (4/4) | +0.090 to +0.240 (4/4) | +0.139 to +0.234 (4/4) | +0.084 to +0.223 (4/4) | +0.325 to +0.394 (4/4) |
| values <= 2.5 keV | 0.995 to 0.998 | +0.097 to +0.162 (4/4) | +0.069 to +0.162 (4/4) | +0.093 to +0.135 (4/4) | +0.055 to +0.138 (4/4) | +0.283 to +0.328 (4/4) |
| values >= 150 eV | 0.909 to 0.914 | +0.083 to +0.161 (4/4) | +0.061 to +0.148 (4/4) | +0.112 to +0.132 (4/4) | +0.073 to +0.116 (4/4) | +0.286 to +0.338 (4/4) |

## incl, a CES sample visible every 10 ms

| rows kept | share of rows | bb vs hist30 | causal vs hist30 | bb vs gp | causal vs gp | bb vs hold |
|---|---|---|---|---|---|---|
| all | 1.000 to 1.000 | +0.009 to +0.024 (2/4) | -0.006 to +0.017 (1/4) | +0.119 to +0.152 (4/4) | +0.092 to +0.144 (4/4) | +0.410 to +0.443 (4/4) |
| all four exclusions | 0.880 to 0.894 | -0.030 to +0.142 (1/4) | -0.044 to -0.017 (0/4) | +0.061 to +0.218 (2/4) | +0.052 to +0.168 (1/4) | +0.101 to +0.150 (1/4) |
| all four exclusions and no row within 100 ms after a blip | 0.830 to 0.841 | -0.037 to +0.139 (1/4) | -0.052 to -0.019 (0/4) | +0.070 to +0.210 (2/4) | +0.049 to +0.182 (1/4) | +0.105 to +0.158 (1/4) |
| jump <= 1 keV | 0.976 to 0.987 | -0.097 to +0.111 (2/4) | -0.118 to -0.060 (0/4) | +0.128 to +0.231 (1/4) | +0.082 to +0.137 (0/4) | -0.218 to +0.022 (0/4) |
| not within 100 ms after a beam blip (absent frame) | 0.940 to 0.945 | +0.009 to +0.023 (2/4) | -0.007 to +0.019 (1/4) | +0.115 to +0.147 (4/4) | +0.088 to +0.138 (4/4) | +0.416 to +0.444 (4/4) |
| not within 30 ms after a low-BES row | 0.977 to 0.982 | +0.012 to +0.022 (2/4) | -0.008 to +0.021 (1/4) | +0.120 to +0.179 (4/4) | +0.093 to +0.174 (4/4) | +0.440 to +0.458 (4/4) |
| only within 100 ms after a beam blip (absent frame) | 0.055 to 0.060 | +0.001 to +0.043 (1/4) | -0.009 to +0.012 (0/4) | +0.116 to +0.264 (3/4) | +0.103 to +0.273 (1/4) | +0.303 to +0.434 (4/4) |
| values <= 2.5 keV | 0.984 to 0.991 | -0.034 to +0.087 (1/4) | -0.056 to -0.048 (0/4) | +0.172 to +0.276 (4/4) | +0.108 to +0.221 (3/4) | +0.171 to +0.229 (4/4) |
| values <= 4 keV | 0.992 to 0.995 | -0.014 to +0.055 (2/4) | -0.033 to -0.013 (0/4) | +0.145 to +0.209 (4/4) | +0.086 to +0.172 (3/4) | +0.270 to +0.344 (4/4) |
| values >= 150 eV | 0.909 to 0.915 | +0.004 to +0.032 (2/4) | -0.005 to +0.024 (1/4) | +0.094 to +0.118 (4/4) | +0.086 to +0.100 (3/4) | +0.379 to +0.430 (4/4) |

## incl, a CES sample visible every 50 ms

| rows kept | share of rows | bb vs hist30 | causal vs hist30 | bb vs gp | causal vs gp | bb vs hold |
|---|---|---|---|---|---|---|
| all | 1.000 to 1.000 | +0.033 to +0.047 (4/4) | -0.001 to +0.043 (3/4) | +0.166 to +0.238 (4/4) | +0.157 to +0.235 (4/4) | +0.403 to +0.420 (4/4) |
| all four exclusions | 0.880 to 0.895 | +0.018 to +0.136 (1/4) | -0.027 to +0.020 (0/4) | +0.067 to +0.117 (2/4) | -0.080 to +0.076 (0/4) | +0.074 to +0.091 (3/4) |
| all four exclusions and no row within 100 ms after a blip | 0.831 to 0.842 | +0.014 to +0.135 (1/4) | -0.031 to +0.008 (0/4) | +0.076 to +0.123 (2/4) | -0.073 to +0.083 (0/4) | +0.067 to +0.089 (2/4) |
| jump <= 1 keV | 0.974 to 0.986 | -0.027 to +0.117 (2/4) | -0.080 to -0.019 (0/4) | +0.025 to +0.189 (2/4) | -0.031 to +0.088 (0/4) | -0.077 to +0.020 (0/4) |
| not within 100 ms after a beam blip (absent frame) | 0.939 to 0.944 | +0.031 to +0.049 (4/4) | -0.002 to +0.047 (3/4) | +0.171 to +0.240 (4/4) | +0.161 to +0.238 (4/4) | +0.404 to +0.421 (4/4) |
| not within 30 ms after a low-BES row | 0.978 to 0.984 | +0.037 to +0.062 (4/4) | -0.002 to +0.045 (3/4) | +0.184 to +0.266 (4/4) | +0.174 to +0.248 (4/4) | +0.423 to +0.433 (4/4) |
| only within 100 ms after a beam blip (absent frame) | 0.056 to 0.061 | +0.017 to +0.068 (2/4) | +0.001 to +0.046 (0/4) | +0.066 to +0.228 (1/4) | +0.038 to +0.207 (1/4) | +0.162 to +0.416 (4/4) |
| values <= 2.5 keV | 0.984 to 0.991 | +0.035 to +0.090 (4/4) | -0.068 to +0.018 (0/4) | +0.088 to +0.237 (4/4) | +0.048 to +0.157 (1/4) | +0.144 to +0.191 (4/4) |
| values <= 4 keV | 0.992 to 0.995 | +0.047 to +0.081 (4/4) | -0.036 to +0.046 (1/4) | +0.113 to +0.198 (4/4) | +0.048 to +0.127 (3/4) | +0.240 to +0.289 (4/4) |
| values >= 150 eV | 0.908 to 0.914 | +0.031 to +0.040 (3/4) | -0.002 to +0.038 (2/4) | +0.163 to +0.199 (4/4) | +0.159 to +0.193 (4/4) | +0.365 to +0.403 (4/4) |

