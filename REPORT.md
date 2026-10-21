# Technical Report: Shipment ETA and Delay Prediction

## 1. Objective

Shipment delays can accumulate across transport legs and handling stops. This project builds an XGBoost-based workflow that predicts overall delay duration, estimates the risk of a material delay, and produces an estimated arrival time with an explanation of influential factors.

The available dataset does not include scheduled arrival times or actual timestamps for individual stages such as port arrival, customs clearance, or final delivery. The model therefore estimates total shipment delay; it does not directly measure stage-by-stage delay propagation.

## 2. Data and target definitions

The supplied synthetic dataset contains 1,000 shipments across 2023 and 24 columns. It includes origin, destination, carrier, shipment type, distance, transit stops, package weight, congestion, weather, shipment date and hour, `delay_hr`, and `delay_bucket`. There are no missing values in the current file.

Two supervised targets are used:

1. **Delay magnitude:** `delay_hr`, treated as a continuous value in hours.
2. **Material-delay risk:** a binary label equal to 1 when `delay_hr >= 6`, otherwise 0.

The six-hour cutoff is an explicit operational assumption. Only nine rows are labeled `on_time`, so a classifier for any delay would be highly imbalanced. The chosen cutoff yields a more useful risk target, though its business meaning should be confirmed.

## 3. Feature design and ETA calculation

The models use route, carrier, shipment type, route type, date and time, distance, number of transit stops, package weight, carrier congestion, weather, temperature, precipitation, storm alerts, and supplied carrier history metrics. Added interactions combine distance with congestion, transit stops with congestion, and precipitation/weather pressure with storm alerts. These features represent plausible logistics relationships available in the dataset.

The dataset has no planned transit duration or scheduled ETA. For a concrete demonstration, planned duration is assumed to be:

```text
planned transit hours = distance_km / assumed route speed + 4 hours * num_transit_stops
```

The assumed speeds are 55 km/h for highway, 40 km/h for mixed, and 30 km/h for rural routes. Estimated ETA is the shipment departure date and hour plus planned duration plus predicted delay. These speed and handling values are illustrative assumptions, not observed schedules.

## 4. Modeling and validation

The data is sorted chronologically and split into 70% training, 15% validation, and 15% test sets: 700, 150, and 150 shipments, respectively. Training data ends on 2023-09-20; validation data ends on 2023-11-11. The latest 150 shipments are held out for final evaluation. Imputation and one-hot encoding are fitted using training data only. Validation data is used for XGBoost early stopping.

The delay regressor is evaluated with mean absolute error (MAE), root mean squared error (RMSE), and R-squared. The material-delay classifier uses a 0.5 probability cutoff and is evaluated with precision, recall, F1, ROC-AUC, and average precision. Scikit-learn is used for preprocessing and metrics; XGBoost is the only predictive model family.

## 5. Held-out results

| Task | Metric | Test result |
|---|---|---:|
| Delay regression | MAE | 1.62 hours |
| Delay regression | RMSE | 1.84 hours |
| Delay regression | R-squared | 0.326 |
| Material-delay classification | Accuracy | 0.72 |
| Material-delay classification | Precision | 1.00 |
| Material-delay classification | Recall | 0.54 |
| Material-delay classification | F1 | 0.70 |
| Material-delay classification | ROC-AUC | 0.937 |
| Material-delay classification | Average precision | 0.962 |

At the 0.5 probability cutoff, the confusion matrix is TN=58, FP=0, FN=42, TP=50. The high-delay share increases from 41.9% in training to 61.3% in the test period. This temporal distribution shift helps explain why the model misses 42 high-delay test shipments at the default cutoff. A decision threshold should be selected using validation data and the relative operational cost of missed delays versus false alarms.

## 6. Explanations

XGBoost contribution values are generated for each held-out prediction. The top global contributors to delay estimates by mean absolute contribution are:

| Feature | Mean absolute contribution (hours) |
|---|---:|
| Historical carrier average delay | 0.883 |
| Rural route indicator | 0.839 |
| Carrier network congestion | 0.351 |
| Storm alert | 0.348 |
| Destination fog | 0.343 |
| Number of transit stops | 0.321 |

Per-shipment explanations are saved in `models/test_predictions.csv`; global rankings are in `models/feature_contributions.csv`. Contribution magnitude describes the model's behavior, not a causal effect. The sign of each local contribution indicates whether that feature moved an individual prediction up or down.

## 7. Limitations and next steps

The data is synthetic, and the test scores do not establish real-world generalization. The provenance of `carrier_avg_delay_hr` and `carrier_on_time_rate` is undocumented. They are treated as historical values available before shipment; if they were calculated from the full dataset, they could leak future target information. Confirm their time availability before relying on the evaluation.

Most importantly, the dataset lacks scheduled and actual timestamps for intermediate stages. To model delay propagation, add stage-level schedule and actual times, derive each stage's delay and remaining transit duration, and predict the final arrival using only information available at the prediction point. Until those fields are available, present this as an overall delay model with an assumption-based ETA.

## 8. Reproduction

From the project root, install dependencies and train both models:

```powershell
python -m pip install -r requirements.txt
python src/train_shipment.py
```

The script writes model files, held-out predictions, evaluation metrics, and contribution summaries to `models/`. The README contains the project setup and output details.
