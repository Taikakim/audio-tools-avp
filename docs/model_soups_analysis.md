# Model Soups Multiprompt Evaluation Analysis

This report analyzes the performance of the different checkpoint averaging weights (**Model Soups**) trained on the `lr 2e-5` run, based on evaluations from `/run/media/kim/Lehto/sa3_control_runs/onset_eval_soups_multiprompt/`.

The evaluation consists of **54 generations per soup** (3 prompts $\times$ 2 seeds $\times$ 3 gains [1.0, 2.0, 3.0] $\times$ 3 target densities [7.0, 8.0, 9.0]).

---

## Executive Summary

* **Timbral Quality Winner**: **`soup_exppeak20_ep10-40`** achieves the highest average Production Quality (PQ = **7.935**) and Content Enjoyment (CE = **6.507**).
* **Stability Winner**: **`soup_exppeak25_ep10-40`** has the lowest disintegration rate overall with only **10 / 54 (18.5%)** collapsed clips (tied with valley).
* **Steering Performance**:
  - At **Gain 1.0**: **`single_ep20_step108k`** has the best steering correlation ($r = 0.963$) and **`soup_exppeak25_ep10-40`** has the highest steering responsiveness (slope = **0.421**).
  - At **Gain 2.0**: **`soup_sine_ep10-40`** maintains the highest steering correlation ($r = 0.958$).
  - At **Gain 3.0**: **`soup_tri_ep10-40`** preserves the best steering correlation ($r = 0.902$), although higher gains generally increase disintegration rates.

## Overall Soup Comparison

| Soup Model | Avg PQ (Acoustic) | Avg CE (Musicality) | Avg CU (Usefulness) | Disintegration Rate (PQ/CE < 6.0) | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`soup_asc_ep10-40`** | 7.813 | 6.256 | 7.763 | **15 / 54 (27.8%)** | Completed |
| **`soup_cosasc_ep10-40`** | 7.779 | 6.175 | 7.725 | **17 / 54 (31.5%)** | Completed |
| **`soup_ep20_ep40`** | 7.784 | 6.258 | 7.739 | **15 / 54 (27.8%)** | Completed |
| **`soup_expasc_ep10-40`** | 7.732 | 6.080 | 7.688 | **20 / 54 (37.0%)** | Completed |
| **`soup_exppeak20_ep10-40`** | 7.935 | 6.493 | 7.866 | **11 / 54 (20.4%)** | Completed |
| **`soup_exppeak25_ep10-40`** | 7.854 | 6.420 | 7.798 | **10 / 54 (18.5%)** | Completed |
| **`soup_sine_ep10-40`** | 7.880 | 6.433 | 7.823 | **12 / 54 (22.2%)** | Completed |
| **`soup_tri_ep10-40`** | 7.860 | 6.425 | 7.810 | **12 / 54 (22.2%)** | Completed |
| **`soup_valley_ep10-40`** | 7.886 | 6.412 | 7.855 | **11 / 54 (20.4%)** | Completed |
| **`single_ep20_step108k`** | 7.931 | 6.507 | 7.857 | **11 / 54 (20.4%)** | Completed |
| **`single_ep25_step135k`** | 7.876 | 6.394 | 7.793 | **13 / 54 (24.1%)** | Completed |
| **`single_ep40_step216k`** | 7.597 | 5.744 | 7.561 | **27 / 54 (50.0%)** | Completed |

---

## Detailed Performance by Gain Level

### 1. `single_ep20_step108k`
| Gain | Avg Correlation ($r$) | Avg Slope | Avg MAE | Avg PQ | Avg CE | Disintegration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0** | 0.963 | 0.321 | 0.69 | 8.07 | 6.83 | 1 / 18 (5.6%) |
| **2.0** | 0.867 | 0.412 | 0.88 | 7.95 | 6.43 | 4 / 18 (22.2%) |
| **3.0** | 0.614 | 0.387 | 0.89 | 7.77 | 6.26 | 6 / 18 (33.3%) |

### 2. `single_ep25_step135k`
| Gain | Avg Correlation ($r$) | Avg Slope | Avg MAE | Avg PQ | Avg CE | Disintegration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0** | 0.914 | 0.317 | 0.71 | 7.99 | 6.71 | 1 / 18 (5.6%) |
| **2.0** | 0.889 | 0.346 | 0.83 | 8.00 | 6.47 | 4 / 18 (22.2%) |
| **3.0** | 0.860 | 0.292 | 1.04 | 7.64 | 6.01 | 8 / 18 (44.4%) |

### 3. `single_ep40_step216k`
| Gain | Avg Correlation ($r$) | Avg Slope | Avg MAE | Avg PQ | Avg CE | Disintegration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0** | 0.755 | 0.229 | 0.81 | 7.95 | 6.34 | 5 / 18 (27.8%) |
| **2.0** | 0.791 | 0.617 | 0.66 | 7.64 | 5.80 | 9 / 18 (50.0%) |
| **3.0** | 0.736 | 0.562 | 0.87 | 7.20 | 5.09 | 13 / 18 (72.2%) |

### 4. `soup_asc_ep10-40`
| Gain | Avg Correlation ($r$) | Avg Slope | Avg MAE | Avg PQ | Avg CE | Disintegration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0** | 0.951 | 0.342 | 0.71 | 7.98 | 6.52 | 3 / 18 (16.7%) |
| **2.0** | 0.504 | 0.446 | 0.79 | 7.90 | 6.33 | 3 / 18 (16.7%) |
| **3.0** | 0.730 | 0.454 | 0.95 | 7.56 | 5.92 | 9 / 18 (50.0%) |

### 5. `soup_cosasc_ep10-40`
| Gain | Avg Correlation ($r$) | Avg Slope | Avg MAE | Avg PQ | Avg CE | Disintegration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0** | 0.419 | 0.237 | 0.79 | 7.98 | 6.49 | 3 / 18 (16.7%) |
| **2.0** | 0.844 | 0.621 | 0.71 | 7.78 | 6.17 | 5 / 18 (27.8%) |
| **3.0** | 0.729 | 0.437 | 0.98 | 7.58 | 5.86 | 9 / 18 (50.0%) |

### 6. `soup_ep20_ep40`
| Gain | Avg Correlation ($r$) | Avg Slope | Avg MAE | Avg PQ | Avg CE | Disintegration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0** | 0.778 | 0.237 | 0.79 | 7.98 | 6.60 | 2 / 18 (11.1%) |
| **2.0** | 0.208 | 0.383 | 0.88 | 7.86 | 6.22 | 4 / 18 (22.2%) |
| **3.0** | 0.726 | 0.417 | 0.97 | 7.51 | 5.96 | 9 / 18 (50.0%) |

### 7. `soup_expasc_ep10-40`
| Gain | Avg Correlation ($r$) | Avg Slope | Avg MAE | Avg PQ | Avg CE | Disintegration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0** | 0.892 | 0.221 | 0.76 | 8.02 | 6.47 | 3 / 18 (16.7%) |
| **2.0** | 0.797 | 0.638 | 0.66 | 7.72 | 6.04 | 7 / 18 (38.9%) |
| **3.0** | 0.821 | 0.633 | 0.87 | 7.45 | 5.73 | 10 / 18 (55.6%) |

### 8. `soup_exppeak20_ep10-40`
| Gain | Avg Correlation ($r$) | Avg Slope | Avg MAE | Avg PQ | Avg CE | Disintegration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0** | 0.941 | 0.283 | 0.70 | 8.07 | 6.80 | 1 / 18 (5.6%) |
| **2.0** | 0.722 | 0.283 | 0.89 | 7.87 | 6.34 | 4 / 18 (22.2%) |
| **3.0** | 0.454 | 0.300 | 0.92 | 7.86 | 6.34 | 6 / 18 (33.3%) |

### 9. `soup_exppeak25_ep10-40`
| Gain | Avg Correlation ($r$) | Avg Slope | Avg MAE | Avg PQ | Avg CE | Disintegration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0** | 0.906 | 0.421 | 0.69 | 7.98 | 6.69 | 1 / 18 (5.6%) |
| **2.0** | 0.915 | 0.275 | 0.87 | 7.98 | 6.50 | 2 / 18 (11.1%) |
| **3.0** | 0.891 | 0.317 | 1.01 | 7.61 | 6.07 | 7 / 18 (38.9%) |

### 10. `soup_sine_ep10-40`
| Gain | Avg Correlation ($r$) | Avg Slope | Avg MAE | Avg PQ | Avg CE | Disintegration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0** | 0.915 | 0.304 | 0.72 | 7.99 | 6.68 | 1 / 18 (5.6%) |
| **2.0** | 0.958 | 0.333 | 0.86 | 7.99 | 6.51 | 3 / 18 (16.7%) |
| **3.0** | 0.893 | 0.346 | 0.91 | 7.66 | 6.11 | 8 / 18 (44.4%) |

### 11. `soup_tri_ep10-40`
| Gain | Avg Correlation ($r$) | Avg Slope | Avg MAE | Avg PQ | Avg CE | Disintegration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0** | 0.914 | 0.371 | 0.68 | 7.98 | 6.68 | 1 / 18 (5.6%) |
| **2.0** | 0.925 | 0.242 | 0.86 | 7.96 | 6.51 | 3 / 18 (16.7%) |
| **3.0** | 0.902 | 0.329 | 0.94 | 7.64 | 6.09 | 8 / 18 (44.4%) |

### 12. `soup_valley_ep10-40`
| Gain | Avg Correlation ($r$) | Avg Slope | Avg MAE | Avg PQ | Avg CE | Disintegration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0** | 0.930 | 0.404 | 0.66 | 8.03 | 6.71 | 1 / 18 (5.6%) |
| **2.0** | 0.510 | 0.300 | 0.85 | 7.95 | 6.41 | 3 / 18 (16.7%) |
| **3.0** | 0.797 | 0.383 | 0.89 | 7.68 | 6.11 | 7 / 18 (38.9%) |
