# Chathouse

Trusted product metrics from raw chat logs. Chathouse turns WildChat-1M conversation logs into silver and gold tables and explores them.

## Layers

| Layer | Content |
|-------|---------|
| Bronze | Raw Parquet files, read as published |
| Silver | `silver_conversations`, `silver_utterances`, `dim_user_proxy` |
| Gold | Daily activity, model mix, engagement, return rates, language and country mix, hourly pattern, user concentration |

## Repository layout

```
Chathouse/
  notebooks/01_data_transformation_and_eda.ipynb   Data transformation and EDA
  requirements.txt                                 Pinned Python packages
  output/gold/                                     Gold tables (written by the notebook, not tracked)
```

## Run

1. Install packages: `pip install -r requirements.txt`
2. Place the WildChat-1M Parquet files in the repository root, or set `CHATHOUSE_DATA_GLOB` to a glob that matches them.
3. Open `notebooks/01_data_transformation_and_eda.ipynb` and run all cells. The notebook reads every file that matches the data path.

## Data

WildChat-1M, Allen Institute for AI. Hugging Face dataset `allenai/WildChat-1M`. Licence: ODC-BY 1.0. Credit the dataset in any output.

## Branches

`main` holds reviewed work. Each user story uses a branch named `feature/<story-id>-<short-name>` and merges by pull request.
