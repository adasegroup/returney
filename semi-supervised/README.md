# Experiments on semi-supervised learning for web user return time prediction
## Articles about semi-supervised Learning for inspiration

#### Semi-Supervised Recurrent Neural Network for Adverse Drug Reaction mention extraction

**link to the article**: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2192-4

Authors  propose a method to extract adverse drug reactions from social media platform's posts. They especially focused on Twitter posts.
Their approach is based on a semi-supervised learning method which operates in unsupervised and supervised phases.
In first phase, they train a Bidirectional-LSTM model to predict the drug name given its context in the tweet:
- Given a tweet, identify the drug name mentions using a curated drug name lexicon. For this phase, they use tweets with exactly one mention of any drug.
- Once drug names are identified in the tweet, replace all drug name mentions with a single dummy token (<_DRUG_>).
- Like in the word2vec training, authors use the context of the masked drug name in the tweet as input to predict the actual drug name.
- Author also utilize name matching systems which handle drugs misspelling.

In the second, supervised phase, authors train *already trained* in the previous phase bi-LSTM network to label words in an input sequence with ADR membership tags. Such initial weights, according to the authors, helps to achieve good results in ADR labels prediction (supervised) training.
So, during prediction (testing) stage, the network weights obtained as result of training on both tasks are used. The overall system pipeline is described in the figure below.
![](https://media.springernature.com/full/springer-static/image/art:10.1186/s12859-018-2192-4/MediaObjects/12859_2018_2192_Fig1_HTML.png?as=webp)

We also can utilize a similar algorithm for user returning time prediction. We can train our model to predict one feature from the given set of session describing features even for the non-returning users. This may help to learn better embeddings of the session features on the whole dataset (including non-returning users). And then train a model with weights initialized in such way to predict returning time.

#### Revisiting LSTM Networks for Semi-Supervised Text Classification via Mixed Objective Function

**link to the paper**: https://www.kdd.org/kdd2018/files/deep-learning-day/DLDay18_paper_46.pdf

Authors train bi-LSTM using objective function consisting of 2 parts: for labeled (supervised part) and unlabeled data points (unsupervised part).

The similar method is already utilized by Grob et al., but we can try to use different loss functions for unlabeled data points.


####  Semi-supervised regression: A recent review

**link to the paper**: https://www.researchgate.net/publication/325798856_Semi-supervised_regression_A_recent_review

This article describes and classify different methods for the semi-supervised regression problem. 

From the mentioned approaches, the most appropriate class of methods for our task is the non-parametric class. These methods trying to find relationships directly from the data. Our current idea is to study similarity between labeled data points and on this basis make some conclusions about returning time of non-returned users. For example, we can try to predict order of returning for the non-returned user (which of them return sooner or later beyond the border of the prediction window).	
