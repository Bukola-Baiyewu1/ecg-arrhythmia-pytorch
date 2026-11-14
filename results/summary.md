Mean ± standard deviation over seeds [0, 1, 2]. Test set: MIT-BIH (DS2 records for the patient split).

| Split | Class weighting | Accuracy | Macro-F1 | Recall N | Recall S | Recall V | Recall F | Recall Q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| beat | none | 0.988 ± 0.001 | 0.735 ± 0.005 | 1.00 | 0.82 | 0.97 | 0.77 | 0.00 |
| beat | sqrt_inverse | 0.986 ± 0.003 | 0.726 ± 0.018 | 0.99 | 0.84 | 0.96 | 0.87 | 0.00 |
| patient | none | 0.924 ± 0.004 | 0.353 ± 0.006 | 0.98 | 0.01 | 0.82 | 0.00 | 0.00 |
| patient | sqrt_inverse | 0.837 ± 0.053 | 0.348 ± 0.009 | 0.89 | 0.07 | 0.72 | 0.01 | 0.00 |
