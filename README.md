# Tree Communication Models for Sentiment Analysis (TCMSA)

Code of Tree communication model (TCM) for sentiment analysis.

# Requirement
* python3
* [tensorflow](https://tensorflow.google.cn/) 
* [fold](https://github.com/tensorflow/fold)

# Get data
```bash
cd data/trees
wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
unzip stanfordSentimentTreebank.zip
rm stanfordSentimentTreebank.zip
```

# Revisit our performance & Test
* Specify Model: config.py --> 'model4test'
* Run the script:
```bash
# specify gpu in run.sh
./run.sh test_new.py
```
* If there is `Variable cannot found` error, please change the variable name of checkpoint accordingly. `tools/rename_model.sh` is a script for variable rename of checkpoints:
```bash
# the output model is saved in the same directory as model_to_save: model_to_save_new
rename_model.sh model_to_reloade
```

* In saved_model/, we release our models under 4 settings.

## Train models

* Obtain initial state from tree-LSTM. We take the tensorflow-fold impletation of tree-LSTM following [sentiment.ipynb](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/sentiment.ipynb).
* Specify initial sate in config.py: 'pre_train'
* Start to Train:
```bash
# specify gpu in run.sh
./run.sh train_new.py
```


## Citation
Please cite our ACL 2019 paper:

```bib
@inproceedings{zhang2019tree,
    title = "Tree Communication Models for Sentiment Analysis",
    author={Zhang, Yuan and Zhang, Yue},
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    pages = "3518--3527",
}
```