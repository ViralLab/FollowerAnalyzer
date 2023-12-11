# FollowerAnalyzer

## Overview

FollowerAnalyzer is a tool to analyze the followers of social media profiles. Given an ordered list of a profile's followers and their profile creation dates, FollowerAnalyzer estimates the follow times and assigns an anomaly score to each follower. The follow time estimation method is based on the algorithm suggested by Meeder et al. [1]. For profiles with more than 10,000 followers, the mean follow time estimation error is in the order of hours. The anomaly score assigned by FollowerAnalyzer is computed using the *Sliding Histogram* method introduced by Zouzou and Varol [2]. Higher anomaly scores indicate that the user may be a part of a coordinated group of fake-followers. Additionally, functions to visualize the followers of a social media profile in a *follower map* are also provided. For details on the follower map and the Sliding Histogram refer to the paper [2].

* [Quick start](https://github.com/YZouzou/FollowerAnalyzer/blob/main/QuickStart.ipynb)

## Installation
The code was developped using Python 3.8.10. Run the following command, preferably within a virtual environment, to install the Python dependencies for this package.

`pip install -r requirements.txt`


## Examples
The notebook **QuickStart.ipynb** has an overview of the different utilities of this package. It also contains a tutorial on incorporating the Secim2023 Twitter (X) dataset [3].

## Citation
```
@article{zouzou2023unsupervised,
  title={Unsupervised detection of coordinated fake-follower campaigns on social media},
  author={Zouzou, Yasser and Varol, Onur},
  journal={arXiv preprint arXiv:2310.20407},
  year={2023}
}
```

## References
1. Meeder, Brendan, et al. "We know who you followed last summer: inferring social link creation times in twitter." Proceedings of the 20th international conference on World wide web. 2011.
2. Zouzou, Yasser, and Onur Varol. "Unsupervised detection of coordinated fake-follower campaigns on social media." arXiv preprint arXiv:2310.20407 (2023).
3. Ali Najafi; Nihat Mugurtay; Yasser Zouzou; Ege Demirci; Serhat Demirkiran; Huseyin Alper Karadeniz; Onur Varol, 2022, "#Secim2023: First Public Dataset for Studying Turkish General Election", https://doi.org/10.7910/DVN/QJA1ZW, Harvard Dataverse, V8, UNF:6:HRWM28liWqqmXgfUI4Xu8Q
