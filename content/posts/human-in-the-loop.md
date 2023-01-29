+++
title = "Designing Human-in-the-Loop ML Systems"
date = "2023-01-26T21:00:00+00:00"
tags = ["machine learning", "mlops"]
author = "Jun Yu"
math = "true"
+++

![png](/images/human-in-the-loop/wanaka.png)

As machine learning practitioners, we constantly strive to produce the highest-performing models to achieve the best business outcomes. But model development is only the tip of the iceberg; how well an ML solution performs has to be continuously evaluated on live predictions. When using trained models, we subtly invoke an assumption: that the training data distribution sufficiently approximates the unseen data distribution. Unfortunately, though, this does not always hold.

<!--more-->

## The Case for Model Monitoring

Model monitoring is crucial because the effectiveness of ML models in production degrades over time. This phenomenon is commonly known as data drift, where the data distribution at inference time is meaningfully different from training time. New trends may appear, unexpected confounders can emerge… there could be myriad reasons why the nature of data between training and inference time might differ. As a quick example, textual datasets obtained before 2020 would not mention COVID-19, thus chatbots (trained on only such datasets) handling customer queries might fail to recognize the pandemic as an emergent topic and provide relevant responses.

As long as models are used in production, we have to constantly monitor their performance and appropriately retrain them.

We can observe a model’s performance in production by evaluating its live predictions, and this entails having a set of ground truth labels for these predictions to be compared against. From here, assuming it is a classification problem, we can calculate standard metrics like accuracy, precision, recall, or any other error measure we desire.

## Feedback Loops

A feedback loop refers to the process of obtaining the ground truth labels of live predictions. In many cases, this occurs naturally: a model recommending similar videos to users can be judged based on the clickthrough rate or other engagement metrics. In this example, the feedback loop for the model’s predictions takes a very short time to materialize; in a matter of seconds or minutes, we’ll know whether the user has watched a suggested video and to what extent.

But in other cases, natural feedback loops can also take a long time. Consider a model predicting whether bank transactions are fraudulent. We truly only know how well our model works when the user raises a dispute (or not) within a time window, which could be months.

In my team, we build systems to enable real-time email intent classification as a part of a platform to automate two-way B2B emails for our customers, where appropriate replies are sent based on the intent of the lead’s email. The primary challenge is maintaining a very high prediction accuracy for each intent category, as misclassifying intents could result in inappropriate or tone-deaf replies, eventually causing sullied impressions or lost revenue opportunities.

Whether it be email intent classification or fraud detection, we want to continually evaluate our ML systems and improve them. To achieve this, how can we drastically shorten these feedback loops so that we can be confident that they are working optimally (or not) in production?

## Human-in-the-Loop Machine Learning

We can enlist the help of human annotators here. This is not a new concept; data scientists often spend a significant chunk of their time labeling data for training, and there are even commercial tools that facilitate this, like AWS Mechanical Turk or Scale AI. But at high inference volumes, labeling all predictions can be immensely time-consuming or expensive.

Furthermore, in some cases like intent classification, human perception is ultimately the most reliable source of truth, thus it would only make sense for models to be judged against human-verified labels, provided that these annotators have a good understanding of the task.

At some point, between the competing concerns of speed, costs, and control, it might be worth investing in an in-house annotation process. Our team maintains a simple data annotation platform alongside a small group of contract annotators working shifts around the clock. This allows us to have a fresh supply of ground truth labels for model predictions quickly (usually less than an hour), and more critically, control our classification strategy to balance accuracy and timeliness.

### Using live annotations as live predictions

For most business cases, predictions are rather time-sensitive. But particularly for medium-latency, high-stakes, and moderately subjective tasks, we can use live annotators to “crowdsource” predictions. Specifically, one can consider the approach of sending these tasks to online and available annotators so that they can participate (in combination with ML model predictions) in a collective voting system to produce a final prediction, using the “wisdom of the crowd” to make high-quality classifications. In other words, using live annotations to decide on live predictions.

There lies an obvious tradeoff with this strategy: waiting for more annotators to participate in live tasks increases the accuracy and reliability of the final prediction, but this inevitably also takes more time (assuming you scale your annotation team responsibly alongside task volume). In balancing this time versus accuracy tradeoff, we can decide how we want to assign these tasks to available annotators: how do we prioritize pending tasks, how many annotations are sufficient for each task, what is the cutoff time, how to resolve contentious tasks (tasks that do not reach a consensus). We have full control to tweak any part of the annotation system and remove bottlenecks until a satisfactory steady state is reached.

It is nonetheless noteworthy that a key limitation of this method is that it is not scalable. Although using annotations as predictions might work in low-velocity situations, it is simply not sustainable to continuously ramp up an annotation team proportionally to its task volume (and concomitant responsibilities like onboarding, coaching, quality control, etc.) while maintaining SLAs. In an ML-enabled system, ML models should ultimately be at the forefront of generating accurate predictions.

### Obtaining ground truth labels

We previously discussed the benefit of using human annotations to form ground truth labels for monitoring model performance. Similar to the previous section, what’s interesting is how we derive a sensible task assignment strategy or algorithm. How do we decide how many agreeing annotations are sufficient to form a ground truth? How do we determine which tasks should be labeled first?

For the latter, an active learning approach can be helpful. Active learning is a set of systems where the learner queries the user (an oracle or information source) to label new data points. This type of system thrives in situations where unlabeled data is abundant but manual labeling is expensive. By intelligently querying for new data points, we can get the model to learn with much fewer but more meaningful data points. Thus by its nature, it is very relevant to human-in-the-loop ML systems.

Here, the productionized model is the learner and the oracle is the annotation system. The simplest query approach would be to prioritize annotation for tasks in which the model is less certain; in other words, assign tasks with model predictions of lower confidence scores (prediction probabilities). By obtaining ground truth labels for these tasks first, we can feed these tasks back into the model more quickly for retraining.

We can choose a suitable set of criteria for which tasks are more important. In certain cases, some might prefer to maintain a sense of class balance, in which we can sample for diversity; or if there are tasks relating to more critical clients, we might want to prioritize them instead.

Another approach, which combines the previous section (for medium-latency, high-stakes tasks) and active learning, is to allow the model to send predictions if its confidence for a task is high, but route it to live annotators and use aforementioned consensus methods if the confidence is low.

## Reliable Annotations

### Implementing clear guidelines

High-quality annotations require clear guidelines — these are the instructions we provide to annotators. For a multi-class text classification task, this entails spelling out distinct definitions and a few examples for each class to make the annotation process as objective as possible. Where there is uncertainty, there should be a way to flag these tasks instead of allowing them to be labeled haphazardly.

### Measuring annotator performance

Managing a team of annotators entails monitoring their performance over time. The main intention is twofold:

1. Assurance that we’re paying for high-quality annotations.
2. Understanding how closely individual annotators are adhering to our guidelines.

One way to assess performance is simply to calculate each annotator's prediction accuracy. Assuming we require at least 3 agreeing annotator predictions to form a ground truth for a task, we can calculate of all the tasks that an annotator has worked on in a particular period, how many of his/her predictions are consistent with the ground truth label. Bonus points for implementing a system that minimizes the risk of annotators blindly copying from one another.

Ideally, annotator accuracy should be maintained at a high level over time. If guidelines are changed, we expect a temporary decline in their accuracy as they adjust to new instructions. However, if we observe a consistent drop in accuracy for multiple operators over time, this might suggest that our guidelines (and thus label classes) are not adequately capturing the nature of incoming live tasks — a problem of data drift (specifically concept drift).

### Considering subjectivity in annotations

Indubitably, there is inherent subjectivity in human annotations. When combining multiple annotations to obtain ground truth labels (which they would be assessed upon as discussed in the above section), we may require more than just accuracy to justify whether an annotator is underperforming. Humans are diverse, and ultimately for tasks with a subjective quality (which is why we’d like human annotations in the first place), it would be helpful to consider and measure this layer of subjectivity and explore how they reach their decisions.

Again, let’s use a text classification task as an example. On top of asking annotators for their class prediction, we can also ask: “what percentage of people do you think will select each label?” They can choose a label as their final prediction even though they don’t feel most people will pick it.

Although it takes more time per task, there are a few benefits to the quality of annotations:

1. Annotators will be less likely to misclick or make careless mistakes as they weigh their opinion on how others might relate to the task.
2. Annotators give more honest and nuanced opinions. They’re allowed to give an answer they believe should be correct, even if it might not align with the perceived majority sentiment. This encourages diverse responses (for more complex tasks) and reduces the pressure to conform.
3. We get information about the label expectation of each task, which can help us better synthesize classifications by considering ambiguity.
4. We can devise a way to study annotators’ trustworthiness/honesty by calculating an additional metric beyond inter-annotator accuracy.

Accompanying the fourth point is the [Bayesian Truth Serum](https://wesselb.github.io/assets/write-ups/Bruinsma,%20A%20Bayesian%20Truth%20Serum.pdf), a statistical method that combines each annotator’s actual selected prediction and their expected predictions into a single score in an information-theoretic approach. I won’t dive into the details here, but this provides an insight into how annotators reason with ambiguity, whether there is a non-independent selection occurring in the annotation process, and the information gain for each annotator’s label for a particular task.

### Krippendorff’s alpha

On the dataset level, we can implement statistical quality control as a measure of reliability. [Krippendorff’s alpha](https://en.wikipedia.org/wiki/Krippendorff%27s_alpha) aims to answer the question: “what is the overall level of agreement in my dataset?”. We wish to find out if annotators agree with one another often enough that we can rely on their labels as ground truths. Krippendorff’s alpha is a calculated value between $[-1, 1]$, and generally can be interpreted as such:

- 0.8 - 1: high agreement; reliable dataset to use for training models
- 0.67 - 0.8: likely that some labels are highly consistent and others are not; low reliability
- 0: random distribution
- -1: perfect disagreement

Krippendorff’s alpha can handle incomplete datasets and generalizes to different sample sizes and number of annotators. However, if the expected agreement is high enough (e.g. 95% of annotator predictions are of one class), then Krippendorff’s alpha will stay relatively low no matter how often they agree, and there is no theoretical way to obtain significance thresholds besides bootstrapping.

Its computation can get quite complex, but fortunately, existing Python libraries help calculate this easily (e.g. [disagree](https://github.com/o-P-o/disagree)).

## Closing

I could go on about designing the annotator experience — including workloads and user interfaces, but this post getting too long. This topic is complex and contains many moving parts, but I hope this post helps highlight some salient motivations, practical considerations, and statistical methods for human-in-the-loop ML systems. For further reading, I highly recommend the book [Human-in-the-Loop Machine Learning](https://www.manning.com/books/human-in-the-loop-machine-learning) by Robert (Munro) Monarch for more in-depth coverage. In this post, I referenced relevant chapters in this book for discussions on annotation subjectivity and Krippendorff’s alpha.

In the era of powerful language models, another alternative I have to mention is the use of models like GPT-3 to label or generate synthetic data (various techniques are detailed in [this paper](https://arxiv.org/pdf/2212.10450v1.pdf)). While advances in LLMs have made leaps and bounds in recent years, I would still encourage caution when relying on these tools to obtain ground truth data, particularly for evaluating live predictions. For now, a human-powered annotation system might be worth considering as a performant and customizable way to drastically shorten your feedback loops and monitor models in production.

*Cover image: Wanaka Tree. Source: [Unsplash](https://unsplash.com/photos/KonWFWUaAuk) (cropped)*
