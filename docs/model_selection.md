# SMARTMAINTAIN Model Selection

## Dataset and common configuration

- Dataset: C-MAPSS FD001
- Task: Remaining Useful Life (RUL) regression
- Training sequences: 17,731
- Test engines: 100
- Input features: 25
- Window size: 30 cycles
- RUL cap: 125
- Same preprocessing and evaluation procedure for all three models

## Models evaluated

1. LSTM
2. GRU
3. CNN-LSTM

## Verified test results

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| LSTM | 11.307567 | 14.743578 | 0.874123 |
| GRU | 11.753654 | 16.559836 | 0.841199 |
| CNN-LSTM | 12.243369 | 17.050261 | 0.831654 |

Lower MAE and RMSE, together with higher R², indicate better agreement with the test RUL values.

## Final baseline

LSTM is retained as the SMARTMAINTAIN production baseline. Among the three evaluated architectures under identical conditions, it achieved the lowest test MAE and RMSE and the highest test R².

Production artifacts:

- `final_model/model.keras`
- `final_model/preprocessing.pkl`

- GRU remains an evaluation experiment only.
- CNN-LSTM remains an evaluation experiment only.
- The production LSTM model must not be replaced unless a future deliberate model-selection decision is made.

## Verification

The three saved prediction CSVs were independently evaluated and reproduced their stored metrics within `1e-4`, using the same 100 test-engine identifiers.
