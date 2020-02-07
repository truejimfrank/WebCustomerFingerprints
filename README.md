# Web Customer Fingerprints

### _LDA customer clustering & future purchase prediction_  

_(LDA : latent dirichlet allocation)_  
A data science adventure by Jim Frank  
_E-commerce website data from  [Retailrocket recommender system dataset](https://www.kaggle.com/retailrocket/ecommerce-dataset) ._

---

## Table of Contents
1. [Data Science Goals](#data-science-goals)
2. [The Data](#the-data)
3. [EDA And Data Wrangling](#eda-and-data-wrangling)
4. [LDA Customer Product Clustering](#lda-customer-product-clustering)
5. [Predicting Customer Purchases](#predicting-customer-purchases)
6. [Conclusion](#conclusion)

---

## Data Science Goals

<b>QUESTION:  </b> 
Which customers did not make a purchase, but were likely to make a purchase soon?  

<b>GOALS:  </b> 
1. LDA customer product relationships quantified by latent grouping probabilities
2. Product purchase prediction with basic dataset features
3. Add latent grouping features to the prediction model 

<b>WHY THIS SET OF GOALS?:  </b> 
The relationships and interactions represented by customer/product interactions may be relevant for useful customer predictions. A perfect case for soft clustering. The anonymized product data means unsupervised grouping is hard to analyze. Improving a prediction model with the customer clusters shows their relevance.

![data fingerprint](http://www.datafingerprint.co.uk/dfplogo2right.jpg)

## The Data

### Context

The data has been collected from a real-world ecommerce website. It is raw data, i.e. without any content transformations, however, all values are hashed due to confidential issues.

### Content

Behaviour data consists of 3 event types. Those being **views, add to carts, and transactions.** These product interactions were collected over a period of 4.5 months in 2015.

Here's an example of the raw data:

| timestamp | visitorid | event | itemid | transactionid |
|--|--|--|--|--|
| 2015-06-01 23:02:12 | 257597 | view | 355908 | NaN |
| 2015-06-01 23:50:14 | 992329 | view | 248676 | NaN |

## EDA And Data Wrangling

* <b>1,407,580</b> unique visitors  
* <b>2,756,101</b> total events  
* <b>2,664,312</b> views  
* <b>69,332</b> add to carts  
* <b>22,457</b> transactions  
* <b>0.81%</b> events that are transactions  

The data required filtering to be useable for LDA clustering.

**2 < # product connections < 400**

Using this criteria yields ample quality data.  
* <b>406,020</b> unique visitors
* <b>235,061</b> unique products
* <b>1,587,292</b> events

![user product hist](https://github.com/truejimfrank/WebCustomerFingerprints/blob/master/img/product_hist.png)

<sub><b>Figure: </b> Visitor counts binned on # products interacted with </sub>

## LDA Customer Product Clustering

When customers view and purchase products from a particular category of product, they create the ties that LDA finds and groups together. As we are finding customer fingerprints, first lets group our customers and products into 10 groups. Multi-dimensional scaling is done with Jensen-Shannon distance.

![lda10](https://github.com/truejimfrank/WebCustomerFingerprints/blob/master/img/cat10_ldavis.png)

<sub><b>Figure: </b> 10 cluster LDA visualization </sub>


![lda10score](https://github.com/truejimfrank/WebCustomerFingerprints/blob/master/img/cat10_scores.png)

<sub><b></b> 10 cluster LDA score. Lower perplexity is better. </sub>

In the 2-dimension representation of multi-dimensional space, you can see that 3 of the topics group closely to one another. Perhaps two of the groups are redundant. Let's group to 8 clusters and take a look.

![lda8](https://github.com/truejimfrank/WebCustomerFingerprints/blob/master/img/cat8_ldavis.png)

<sub><b>Figure: </b> 8 cluster LDA visualization </sub>

![lda8score](https://github.com/truejimfrank/WebCustomerFingerprints/blob/master/img/cat8_scores.png)

<sub><b></b> 8 cluster LDA score. Lower perplexity is better. </sub>

I like the look of this. Quality separation in multi-dimensional space. The anonymized product ID's just means we'll have to improve our logistic prediction to prove the worth of this LDA clustering.

## Predicting Customer Purchases

Here are the terms of our prediction: With a logistic regression, predict True/False if a visitor has purchased a product or will purchase a product in the near future. The time_hour feature attempts to track customer activity. It is number of hours between earliest and latest events.

| product_count | addtocart | view | time_hour |
|--|--|--|--|
| integer | integer | integer | float |

<sub><b>Table: </b> Customer features used for basic logistic regression. </sub>

Data was appropriately sorted, split, and standardized before training the logistic regression predictor.

| X1 | X2 | X3 | X4 |
|--|--|--|--|
| 0.87 | 77.48 | 1.32 | 1.02 |

<sub><b>Table: </b> Feature coeffs contributing to odds for purchase </sub>

| 0.928 acc |  | predicted yes |
|--|--|--|
|  | 95923 | 7118 |
| actual yes | 1122 | 11077 |

<sub><b>Table: </b> Train confusion matrix </sub>

| 0.925 acc |  | predicted yes |
|--|--|--|
|  | 23974 | 1786 |
| actual yes | 795 | 7792 |

<sub><b>Table: </b> Test confusion matrix </sub>

Predictions are doing reasonably well for how unsophisticated this model is. Especially considering that it is predicting mainly from one feature, addtocart.

### Adding LDA Features

| X1 | X2 | X3 | X4 | X5 | X6 | X7 | X8 | X9 | X10 | X11 | X12 |
|--|--|--|--|--|--|--|--|--|--|--|--|
| 0.91 | 76.14 | 1.21 | 1.02 | 0.11 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 |
| product_count | addtocart | view | time_hour |

<sub><b>Table: </b> Feature coeffs contributing to odds for purchase </sub>

The added features are uniformly ineffective.

## Conclusion

Today we've only had time for one cluster model paired with one regression classifier. Thankfully, there are many other algorithymic options to choose from in the data science toolbelt.
