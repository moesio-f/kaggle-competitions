# [Predict the Introverts from the Extroverts](https://www.kaggle.com/competitions/playground-series-s5e7)

> Welcome to the 2025 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.
>
> Your Goal: Your objective is to predict whether a person is an Introvert or Extrovert, given their social behavior and personality traits.

## Timeline

- Start Date - June 30, 2025
- Entry Deadline - Same as the Final Submission Deadline
- Team Merger Deadline - Same as the Final Submission Deadline
- Final Submission Deadline - July 31, 2025


## Process & Methodology

1. **Exploratory Data Analysis**: data understanding, validation, feature engineering (data for experimentation);
2. **Prototype**: scoring metrics definition, prototype of 3-5 algorithms/approaches;  
3. **Prototype Evaluation**: decide on which approach to move forward;
4. **Development**: improve solution (i.e., feature engineering), and make simple API for model serving;

## Structure

- [`notebooks`](./notebooks): experimental notebooks;
    - All notebooks were used as foundation to construct a solution for the problem;
- [`playground_series_s5e7`](./src/playground_series_s5e7): final solution for the competition;
- [`data`](./data): placeholder for datasets;
    - [`competition`](./data/competition): competition data;
    - [`prototype`](./data/prototype): data used for prototype evaluation (i.e., notebooks);
    - [`processed`](./data/processed): training data used for the model (versioned by `v1`, `v2`, etc);
