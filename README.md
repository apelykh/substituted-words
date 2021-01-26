# Substituted words

<p align="center">
  <img src="https://imgs.xkcd.com/comics/horse.png" />
</p>

## Task description

In a given tokenized sentence, some tokens might be replaced with a randomly picked word.

For example:

```
# Original sentence:
the cat sat on the mat

# Two words replaced:
the cat apple on done mat
```

The task is to predict which words were substituted. Specifically, for each token in a sentence, a probability of replacement should be found:

For example:

```
# Input:
the cat apple on done mat

# Output:
0.1 0.2 0.89 0.1 0.99 0.3
```


## Approach

In the current project, the task is approached as a binary token classification where the replacement probability for each token is directly
predicted by the model.
Two model architectures for token classification are explored in this project:

  1. Embedding -> BLSTM -> dropout -> linear layer -> softmax
  2. Pre-trained BERT encoder -> dropout -> linear layer -> softmax

Embedding layer in the LSTM architecture is initialized with 100-d [GloVe vectors](https://nlp.stanford.edu/projects/glove/),
pre-trained on Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased) that are tuned throughout the whole training process.
The embedding layer is followed by a 1-layer BLSTM with hidden size 200, the outputs of which are passed to a dropout layer and a subsequent
linear layer with 2 output neurons.

A [BertForTokenClassification](https://huggingface.co/transformers/model_doc/bert.html?highlight=bertfortokenclassification#bertfortokenclassification)
implementation from [Hugging Face](https://huggingface.co/) in its *base-uncased* configuration is used as a core of the second explored model.
A pre-trained [BertTokenizer](https://huggingface.co/transformers/model_doc/bert.html?highlight=berttokenizer#transformers.BertTokenizer) is used for
data processing while [TokenClassificationPipeline](https://huggingface.co/transformers/main_classes/pipelines.html?highlight=tokenclassificationpipeline#transformers.TokenClassificationPipeline)
takes care of the model inference.

Both architectures are trained on **train** data subset with *CrossEntropyLoss*, *AdamW* optimizer and a linear learning rate scheduler.


## Running inference
**LSTM inference:**

[LSTM trained weights](https://drive.google.com/file/d/1yMbI4XNHF1lY0uKMeSUBzJnYCScKjUaQ/view?usp=sharing) should be downloaded into *./weights*

[LSTM word2id](https://drive.google.com/file/d/18eH5mbRtD8DDLR345k46bqbuz8Su7_ws/view?usp=sharing) should be downloaded into *./assets*

    $ python lstm/inference.py --src_file ./data/val.src --results_file ./data/val.scores.lstm


**BERT inference:**

[BERT trained weights](https://drive.google.com/file/d/1wsodiKWNmcponvVttC5OQ4v-h47daEwg/view?usp=sharing) should be unpacked to *./weights*

    $ python bert/inference.py --src_file ./data/val.src --results_file ./data/val.scores.bert


## Evaluation

For evaluation, the predicted probabilities are first converted into hard labels. Values < 0.5 correspond to the negative class (no substitution
made at the position), everything else is positive.

The metric used for evaluation is F0.5 score which combines precision and recall of the predictions into a single number between 0 and 1 in the following way:
```
F0.5 = 1.25 * p * r / (0.25 * p + r)
```
where p - precision, r - recall. The higher the score, the better.

To run evaluation:

    $ python eval.py submission.txt --golden gt_labels.txt


## Results

The following results on **val** subset were achieved:

 metric | LSTM | BERT
------- | ---- | ---- 
FP | 56864 | 4838
FN | 4118 | 1049 
TP | 18380 | 21449 
TN | 174507 | 226533 
Precision | 0.244 | 0.816
Recall | 0.817 | 0.953 
F0.5 | **0.284** | **0.84** 


## Discussion and further work
