# Data Science Applied to Kaggle Competitions

This repository contains a collection of solutions I came up with for some Kaggle competitions. The mais goal of this repository is to serve as a collection of organized solutions for competitions. In general, the process/methodology mimics the ones I'd apply to a real-world problem (i.e., planning, prototyping, solution development), except for deployment and monitoring. 

The code is made publicly available after the competition submission deadline. For playground competitions, the code is made available as I progress.

## Structure

The repository is organized with the following convention:

```
.
├── <Competition A>/
│   ├── notebooks/
│   │   └── <files>.ipynb
│   ├── data/
│   │   └── ...
│   ├── src/
│   │   └── <competition_id>/
│   │       ├── __init__.py
│   │       └── ...
│   ├── README.md
│   └── pyproject.toml
└── <Competition B>/
    ├── notebooks/
    │   └── <files>.ipynb
    ├── data/
    │   └── ...
    ├── src/
    │   └── <competition_id>/
    │       ├── __init__.py
    │       └── ...
    ├── README.md
    └── pyproject.toml
```

Every competition directory is self-contained, and might use different Python/libraries versions.

## Competitions List

| Competition | Status | Public LB | Private LB | Rank #1 | %Change |
| --- | --- | --- | --- | --- | --- |
| [Predict the Introverts from the Extroverts](./Predict%20the%20Introverts%20from%20the%20Extroverts) | Closed | `0.973279` | `0.967813` | `0.970647` | `-0.292%` |
| Binary Classification with a Bank Dataset | Ongoing | _TBA_ | _TBA_ | _TBA_ | _TBA_ | _TBA_ |
| NeurIPS - Ariel Data Challenge 2025 | Ongoing | _TBA_ | _TBA_ | _TBA_ | _TBA_ | _TBA_ |
