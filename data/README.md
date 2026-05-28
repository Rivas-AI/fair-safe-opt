# Data

Raw, cached, and processed datasets are not committed.

The Adult Income dataset should be retrieved through scikit-learn with:

```python
sklearn.datasets.fetch_openml(data_id=1590, as_frame=True)
```

Tests and smoke checks must use synthetic data instead of downloading Adult.
