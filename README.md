# PipelineModel
## Execution
``pip install -r requirements.txt``

1. generate training data

``$ mkdir training_data; python scripts/gen_training_data.py data/train_weight_first_weight_20``

2. train model

``$ python scripts/train.py training_data/data_2.json``
