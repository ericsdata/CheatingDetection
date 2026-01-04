## Cheating Challenge

#### Read through notes

- Costs
    - False Negatives (missing confirmed cheating) should have highest penality
    - False Positives (lead to blocking of candidates)
    - Manual Review has a small cost
    - 

    > False Negative (cheating passes through): $600
    > False Positive in auto-block region: $300
    > False Positive in manual review region: $150
    > True Positive requiring manual review: $5
    > Correct auto-pass or auto-block: $0`

- Decision Regions
    - Autopass (low cheating risk)



## Plan

- EDA
     - XGB Shap Analysis
     - Charts
- Build predictive model
- Explore network graph
    - Network graph embeddings
- Build out cost based framework
    - Eval function determines optimal decision thresholds that minimize cost across three regions
    