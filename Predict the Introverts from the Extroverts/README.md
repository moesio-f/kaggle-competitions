# [Predict the Introverts from the Extroverts](https://www.kaggle.com/competitions/playground-series-s5e7)

> Welcome to the 2025 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.
>
> _Your Goal:_ Your objective is to predict whether a person is an Introvert or Extrovert, given their social behavior and personality traits.

## Timeline

- Start Date - June 30, 2025
- Entry Deadline - Same as the Final Submission Deadline
- Team Merger Deadline - Same as the Final Submission Deadline
- Final Submission Deadline - July 31, 2025


## Process & Methodology

1. **Exploratory Data Analysis**: data understanding, validation, feature engineering (data for experimentation);
    - Notebooks `00` to `02` contains this stage;
2. **Prototype**: scoring metrics definition, prototype of 3-5 algorithms/approaches;  
    - Notebooks `03` to `06` contains the prototypes with simple algorithms (e.g., SVM, DT, KNN, RF);
3. **Prototype Evaluation**: decide on which approach to move forward;
    - Given the results, SVM and KNN obtained the highest accuracy;
    - KNN had a slight advantage (0.11% gain) over SVM;
    - SVM has more parameters and learns a model from the data;
    - KNN stores the underlying dataset but allows for domain-knowledge (e.g., custom distances and weight-schemes);
    - The final decision is to move forward with **SVM**, due to the expectation that it should be easier to tune, and it allows for easy retraining while keeping the hardware requirements "constant" (i.e., if 10k+ more samples become available KNN memory consumption would grow);
4. **Development**: improve solution (i.e., feature engineering), and make simple API for model serving;
    - The final solution is available at [`playground_series_s5e7`](./src/playground_series_s5e7);
    - The package has the following structure:
        - `feature_engineering`: feature engineering strategies;
        - `training`: model training and tuning;
        - `inference`: model retrieval and inference;
        - `fallback`: model-agnostic decision logic;
        - `api`: wrapper for an API for model serving;
            - RESTFul API and SDK;

## Structure

- [`notebooks`](./notebooks): experimental notebooks;
    - All notebooks were used as foundation to construct a solution for the problem;
- [`playground_series_s5e7`](./src/playground_series_s5e7): final solution for the competition;
- [`data`](./data): placeholder for datasets;
    - [`competition`](./data/competition): competition data;
    - [`prototype`](./data/prototype): data used for prototype evaluation (i.e., notebooks);
    - [`processed`](./data/processed): training data used for the model (versioned by `v1`, `v2`, etc);
