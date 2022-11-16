## Use case - Fraud detection with [Intel® Distribution of Modin](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-of-modin.html#gs.7hvt6) 


This tutorial provides a use case where a credit card company might benefit from machine learning techniques to predict fraudulent transactions. This is the first part of a three-part series.

In this post, you’ll prepare and pre-process data with the [Intel® Distribution of Modin*] (https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-of-modin.html#gs.7hvt6). You’ll also use an [anonymized dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) extracted from Kaggle.

The Intel® Distribution of Modin will help you execute operations faster using the same API as [pandas](https://pandas.pydata.org/). The library is fully compatible with the [pandas API](https://pandas.pydata.org/docs/reference/index.html). OmniSci powers the backend and provides accelerated analytics on Intel® platforms. (Here are [installation instructions](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-of-modin.html#gs.8blx9q).) 

Note: [Modin](https://modin.readthedocs.io/en/stable/) does not currently support distributed execution for all methods from the pandas API. The remaining unimplemented methods are executed in a mode called “default to pandas.” This allows users to continue using Modin even though their workloads contain functions not yet implemented in Modin. 

<img src="https://www.intel.com/content/dam/develop/public/us/en/images/diagrams-infographics/diagram-modin-arch-16x9.jpg.rendition.intel.web.1072.603.jpg"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

## Section 1 : Pre-process and Initial Data Analysis

The first step is to pre-process the data. After you download and extract the data, you’ll have it in spreadsheet format. That means you’ll work with tabular data where each row is a transaction (example) and each column is a feature (transaction amount, credit limit, age.) In this tutorial you won’t know which represents each feature, since the data has been anonymized for privacy purposes. This example uses supervised learning, meaning the algorithm is trained on a pre-defined set of examples. The examples are labeled with one column called LABEL (FRAUD or NOT FRAUD) 

```
t0 = time.time()
pandas_df = pandas.read_csv("creditcard.csv")
pandas_time = time.time()- t0

t1 = time.time()
modin_df = pd.read_csv("creditcard.csv")
modin_time = time.time() - t1

print("Pandas Time(seconds):",pandas_time,"\nModin Time(seconds):",modin_time)
verify_and_print_times(pandas_time, modin_time)
outputDict={"Pandas":pandas_time,"Modin":modin_time}
plotter(outputDict)
```

![Pandas](/tutorials/Optimized Fraud Detection/Images/features.png)

###Dealing with missing values

One of the first steps when analyzing the data is "Missing values". Sometimes the data that we have available is extracted from other sources which could imply that a system can fail to report data in a particular time what would be a missing value in our dataset, is nobody fault it's just that we are working with machines and sometimes they could fail ;). When we face an scenario like this is important to know what to do with missing values (OTHER TUTORIAL).
Let's see then if we have missing values in our data. 

```
t0 = time.time()
print(pandas_df.columns[pandas_df.isna().any()])
pandas_time = time.time()- t0

t1 = time.time()
print(modin_df.columns[modin_df.isna().any()])
modin_time = time.time() - t1
```
![na](/Users/emlanza/Library/CloudStorage/OneDrive-IntelCorporation/Technical/S2E/Content/Images/na.png)

**Modin benefit XXX**

GREAT! No missing values so let's move to the next part.

###Subsampling

In the next part we will just use modin_df in the following examples to avoid duplicated (full code is available HERE). 

Let's now take a look of the distribution of our data.

```
sub_sample_plot=sns.countplot(pandas_df["Class"])
sub_sample_plot
```

![Image](/Users/emlanza/Library/CloudStorage/OneDrive-IntelCorporation/Technical/S2E/Content/Images/distribution.png)

As we can see the class (FRAUD or NO FRAUD) is very UNBALANCED. What means that most cases aren't fraud and just a few are FRAUD. If we would like to train a model with the entire data that we have, the model will learn to detect the majority of the cases (NO FRAUD),which is not what we want, WE WANT TO DETECT FRAUD! If a model is trained with this data it would be able to get high levels of accuracy, but this is not the

There are some techniques to help us on that.

1. Get more FRAUD examples : We should ask to the person whi provide us the dataset to get more examples. It could be applicable in some scenarios but in most of them we have to do what we can with the data we have.
2. Augment FRAUD examples : If there are examples of the class that we want to detect, we could use an algorithm to create sintetic data to get a considerable ammout of examples of the desired class. This technique is used manly in computer vision scenarios but it could be used in any other scenarios.(Will be explanied in other tutorial)
3. Get a new dataset where the ratio fraud vs non-fraud could be close to 1:1

Let's try then to create a new data set with a ratio that could be useful to make the algorithm able to generalize both classes.

It's always a good idea to normalize the data to reduce some effect that an outlier or the variance of the data could cause. There are different alternatives to normalize (the idea is to have the data represented in values from 0 an 1, or -1 and 1 depending on data). Sklearn give us a tool a normalize it easily.
In this part the features "Time" and "Amount", need to be normalized becuase their values are completly different to other features and it can affect the model (We'll go deep in part 2 of the tutorial)

```
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
%time
scaler = StandardScaler()
robust = RobustScaler()
scaled_time=robust.fit_transform(modin_df["Time"].values.reshape(-1,1))
scaled_amount=robust.fit_transform(modin_df["Amount"].values.reshape(-1,1))

```
BEFORE

![normal](/Users/emlanza/Library/CloudStorage/OneDrive-IntelCorporation/Technical/S2E/Content/Images/normal.png)

AFTER

![scaled](/Users/emlanza/Library/CloudStorage/OneDrive-IntelCorporation/Technical/S2E/Content/Images/scaled.png)

**MODIN Benefit xxx**

Let's now create the NEW balanced dataset. 

```
modin_df1_scaled = modin_df1_scaled.sample(frac=1)  #Shuffling the dataframe

modin_df1_scaled_nf = modin_df1_scaled.loc[modin_df["Class"] == 0][:500]
modin_df1_scaled_f = modin_df1_scaled.loc[modin_df["Class"]==1]

# Will reuse all fraud points, will random sample out 500 non-fraud points

# New sample Table
modin_df1_distributed = pd.concat([modin_df1_scaled_nf,modin_df1_scaled_f])
modin_df2 = modin_df1_distributed.sample(frac=1, random_state=42)

modin_df2.head()
```


We can see now that the dataset is now balanced and it will help us to train our algorithm better.

```
sub_sample_plot=sns.countplot(df2["Class"])
sub_sample_plot

```

![balanced](/Users/emlanza/Library/CloudStorage/OneDrive-IntelCorporation/Technical/S2E/Content/Images/balanced.png)


##***2ND PART ANALYTICS OF DATA***.

* Analsys of outliers
* Normalization
* Correlation of data

##***3RD TRAINING***.

* Sklearn intel optimized
* Explanation about perfomance and metrics 



**Let's go!**



## Comments

GOAL : Show the benefit of using intel optimized toolkits with a very easy example (Not much technical details). The example can be a real case challenge for some developers like in this example is Fraud detection, future cases could be computer vision and NLP systems. A complete series of Hugging face algorithms could be also interesting.
