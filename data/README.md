# Dataset

`synthetic_shipment_data.csv` is the shipment dataset used by this project. It contains 1,000 rows and 24 columns covering route, carrier, shipment type, distance, transit stops, weather, congestion, shipment time, delay hours, and delay bucket. The current file has no missing values.

The label `delay_hr` is the regression target. The project defines a material/high delay as `delay_hr >= 6` hours for classification. The file does not include scheduled arrival times or actual timestamps for intermediate stages; see the root README and report for the ETA assumption and limitations.
