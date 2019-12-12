# Detecting Unpalatable Questions on Reddit

Welcome! In this project, we aim to detect unpalatable questions in online discourse. This repo contains the annotated data and source code for all the learning models and analyses.

## Code
The ```/src/``` folder contains the code for:
- deep learning models in ```/deep-learning/```
- traditional machine learning models and all the feature categories in ```/traditional-machine-learning/```
- crowdsourcing annotations using Mechanical Turk API in ```/mturk/```. Check out these <a href="https://docs.google.com/presentation/d/1hsLhWTNGkFxmvQnJk_qhdbn6uqH8MVno-vkZage7ZC4/edit?usp=sharing">slides</a> for an introduction to MTurk and our workflow design.
- comparison of the two question filtering approaches: rule-based regex (vs) constituency parsing in ```Question-Filter-Comparison.ipynb```
- annotator agreement measures in ```Annotation-Quality.ipynb```
- Perspective API toxicity classifier's performance on our dataset in ```/perspective-api-analysis/```


## Data
The file ```/data/annotations_UQ.csv``` contains 10,909 Reddit comments annotated for whether they contain an unpalatable question or not.

```python
# To read as a DataFrame:
import pandas as pd
df = pd.read_csv('/data/annotations_UQ.csv', lineterminator='\n')
```

Column description:
- ‘reply_id’ = unique ID for each row
- ‘reply_text’ = text for the main comment (or reply)
- ‘comment_text’ = text for the preceding comment in the thread
- ‘label’ = majority label selected by MTurk coders. It can take two values: “yes_unpalatable” or “not_unpalatable”
- ‘confidence’ = annotator agreement. Since we collect five annotations, it can take three values: 0.6, 0.8, 1.0
- Note that a very small number of comments received more than five annotations since they were dynamically used as test questions across batches, and confidence values for those rows are not exactly 0.6, 0.8, or 1.0. This is captured in the column ‘unmodified_confidence’. We created the ‘confidence’ column from ‘unmodified_confidence’ using the following brackets: [0.5, 0.7) -> 0.6 ; [0.7, 0.9) -> 0.8 ; [0.9, 1.0] -> 1.0
