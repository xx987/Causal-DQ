# Causal Deep Q-Network (Causal DQ)

This repository provides the implementation associated with the paper:  
**_Causality-informed Anomaly Detection in Partially Observable Sensor Networks: Moving Beyond Correlations_**.

## Usage Instructions

1. Run `CausalDQ_xx_train.py` to start training the network for the corresponding simulation case.  
2. Run `Test_CausalDQ_xx.py` to reproduce the corresponding simulation results.
3. We use the real datasets, Tennessee Eastman Process (TEP) [1] and Solar Flare Detection (SFD) [2],  in the real cases.
## Reminder

If you wish to experiment with different values of `p` (number of variables) or different levels of mean shift,  
**please restart the kernel or the entire code system** to ensure the scenarios are refreshed properly.


## References

[1] Ahmadzadeh, A., Aydin, B., & Kempton, D. (2020). BigData Cup Challenge 2020: Flare Prediction. [Kaggle](https://www.kaggle.com/competitions/bigdata2020-flare-prediction)

[2] Rieth, C. A., Amsel, B. D., Tran, R., & Cook, M. B. (2017). Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation. _Harvard Dataverse_, https://doi.org/10.7910/DVN/6C3JR1

