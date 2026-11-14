Mean ± standard deviation over seeds [0, 1, 2]. Test set: MIT-BIH (DS2 records for the patient split).

| Split | Input | Class weighting | Accuracy | Macro-F1 (5 classes) | Macro-F1 (N,S,V,F) | Recall N | Recall S | Recall V | Recall F | Recall Q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| patient | beat + RR | none | 0.924 ± 0.013 | 0.363 ± 0.014 | 0.453 ± 0.018 | 0.97 | 0.04 | 0.88 | 0.00 | 0.00 |
| patient | beat + RR | sqrt_inverse | 0.834 ± 0.067 | 0.371 ± 0.023 | 0.464 ± 0.028 | 0.86 | 0.32 | 0.91 | 0.03 | 0.00 |
