# The Legal Argument Reasoning Task in Civil Procedure

We present a new NLP task and dataset from the domain of the U.S. civil procedure. Each instance of the dataset consists of a general introduction to the case, a particular question, and a possible solution argument, accompanied by a detailed analysis of why the argument applies in that case. Since the dataset is based on a book aimed at law students, we believe that it represents a truly complex task for benchmarking modern legal language models. 

This repository provides the code for creating the corpus, and training the evaluated models introduced and described in the following paper:

```plain
@InProceedings{Bongard.et.al.2022.NLLP,
  title     = {{The Legal Argument Reasoning Task in Civil Procedure}},
  author    = {Bongard, Leonard and Held, Lena and Habernal, Ivan},
  booktitle = {Proceedings of the Natural Legal Language Processing
               Workshop 2022},
  pages     = {(to appear)},
  year      = {2022},
  address   = {Abu Dhabi, UAE}
}
```
## Obtaining the dataset

The corpus can be obtained by filling out the [request form](/forms/Legal-Argument-Reasoning-Request-Form.docx?raw=true)  and sending a PDF to Dr. Ivan Habernal ( ivan.habernal@tu-darmstadt.de ).

### Terms of use
We wish to be able to share the data set with the research community (“users”) under the following conditions:

1. The dataset is available under the following conditions:
2. The dataset is to be used for non-commercial purposes only.
3. All publications on research based on the dataset should give credit to the author and the publisher.
4. No part of the dataset may be shared with third parties. The dataset may only be used by the person who agrees to the terms of the license.
The dataset must be deleted after finishing the experiments.

## Requirements
The following command installs all necessary packages:

~~~
pip install -r requirements.txt
The project was tested using Python 3.8.
~~~ 


To recreate the corpus a virtual copy of the book "The Glannon Guide to Civil Procedure" (Forth Edition) written by Joseph W. Glannon is required.



## Task
Given a question with a possible correct answer and a short introduction to the topic of the question, identify if the answer candidate is correct or incorrect.

## Train Models
To train the model based on the task simply run:

~~~
python train_and_evaluate_transformer_classifier.py --dataset_type "keep_question"  --model "nlpaueb/legal-bert-base-uncased"
~~~

