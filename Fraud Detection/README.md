This tutorial provides a use case where a credit card company might benefit from machine learning techniques to predict fraudulent transactions. This is the first part of a three-part series.

In this post, you’ll prepare and pre-process data with the Intel® Distribution of Modin*. You’ll also use an anonymized dataset extracted from Kaggle*.

The Intel® Distribution of Modin will help you execute operations faster using the same API as pandas*. The library is fully compatible with the pandas API. OmniSci* powers the backend and provides accelerated analytics on Intel® platforms. (Here are installation instructions.) 

Note: Modin does not currently support distributed execution for all methods from the pandas API. The remaining unimplemented methods are executed in a mode called “default to pandas.” This allows users to continue using Modin even though their workloads contain functions not yet implemented in Modin. 