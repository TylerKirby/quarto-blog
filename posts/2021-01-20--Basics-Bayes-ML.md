---
aliases:
- /machine_learning/2021/01/20/-Basics-Bayes-ML
categories:
- machine_learning
date: '2021-01-20'
description: Bayesian methods can offer capabilities like uncertainty estimates and
  encoding domain knowledge directly into a model. This post provides an overview
  of Bayesian methods beginning with a review of probability before showing how it
  can be applied to the coin flipping problem. By the end, the reader will have a
  basic understanding of the methods and where to go from here.
layout: post
title: The Basics of Bayesian Machine Learning
toc: true

---

If you've ever been dazzled, or perhaps terrified, by dense integral equations and 
perplexing terms such as *marginal log likelihood* and *posterior inference* when 
reading a new research paper, you are certainly not alone. For the uninitiated,
Bayesian methods appear relegated to austere math-magicians at the heights of the 
ivory tower conjuring new results with arcane techniques. Fortunately, the fundamentals
of Bayesian machine learning are much simpler than they appear and rightly so lest we forget 
their origins in 18th century Parisian gambling halls at the hands of one Pierre-Simon, marquis de Laplace. In this short
post, I'll outline Bayes' theorem and how it lies at the heart of all methods in Bayesian
machine learning. Internalizing and understanding this basic theorem is
all that is required to become a practitioner of Bayesian methods. I will also
touch upon some major problems of these methods such as prior elicitation and
intractable posteriors. To conclude I will point to some open problems of interest in
the field and point to more complete resources for mastering Bayesian machine learning.

# Reviewing the Basics
Complete courses in statistics, probability, linear algebra, and calculus are
required indeed to fully understand and appreciate the nuance and elegance of 
Bayesian methods, but the basic laws of probability theory are really all that's
needed to approach Bayes' theorem. We'll review them briefly below.

The **Product Law** allows us to compute the probability of two events occuring
concurrently:

$$Pr(A \cap B) = Pr(B | A) \cdot Pr(A)$$

Note the term $Pr(B | A)$ in the equation above. This can be read as "the probability of $B$ given $A$" and is
called a **conditional probability**. As we shall soon see, Bayes' Theorem is a formulation of
conditional probability.

The **Sum Law** allows us to compute the probability of one event if we know the joint probabilities between it
and another event:

$$Pr(A) = \sum_{i} Pr(A \cap B_i)$$

We assume that the events $B$ can be divided into partitions where we know the joint probability of $A$ and the 
partition $B$. For example, suppose $Pr(A)$ is the probability of flipping heads on a two-sided coin and $Pr(B)$
is the probability of me using a fair coin. Let $Pr(B_1)$ be the probability that the coin is fair and $Pr(B_2)$
be the probability that the coin is unfair. Thus, we could compute the unknown $Pr(A)$ as $Pr(A \cap B_1) + Pr(A \cap B_2)$
assuming that we know the joint probabilities of events $A$ and $B$, or, more simply, assuming we know the
effect of an unfair coin on the probability of flipping heads. You may see how we can generalize the Sum Law 
to handle continuous cases by integrating over the partitions of $B$ rather than summing.

# Bayes' Theorem
With the two laws outlined above, we're ready to tackle **Bayes' Theorem**. Let's begin with the simplest formulation
of the theorem:

$$Pr(B|A) = \frac{Pr(A|B) Pr(B)}{Pr(A)}$$

The numerator is just the Product Law applied to the joint probability of $A$ and $B$. Solving for $P(B|A)$ in the
Product Rule yields Bayes' Theorem above. We will need to go further, however, for this to be useful. In practice,
finding $Pr(A)$ is really hard, so it is more common to substitute it with the result from the Sum Law:

$$Pr(B|A) = \frac{Pr(A|B) Pr(B)}{\sum_{i} Pr(A \cap B_i)}$$

We should generalize the expression for continuous variables since those will often be the variables provided. So far we have used discrete events
$A$ and $B$ to illustrate the results. Let's introduce continuous variables $a$ and $b$ for our generalized expression:

$$p(b|a) = \frac{p(a|b) p(b)}{\int p(a|b) p(b) db}$$

The process of integrating $p(a|b) p(b)$ to find $p(a)$ is called **marginalizing** over the parameter $b$ and
is extremely common in practice.

Each part of Bayes' Theorem has a specific name and function. The probability $p(b|a)$, which is the result
we're seeking to compute, is the **posterior probability** or belief. It is *posterior* from the Latin for *after* since
it is the result we arrive at after calculating with Bayes' Theorem. The term $p(a|b)$ is the **likelihood** since
it represents the likelihood of $a$ given $b$. The probability of $b$ is the **prior probability** or belief. We
begin the Bayesian process with some idea of what we think the prior $p(b)$ is and after incorporating the 
new evidence presented in the likelihood $p(a|b)$ we arrive at a new belief of $p(b)$ weighted against the new evidence.
Finally, we have the ugly term $\int p(a|b) p(b) db$ which is our **marginal probability** or belief. Note that in
each definition I suggest probability and belief can be used interchangeably. In the Bayesian context, they most
certainly can though this can get you into trouble mathematically elsewhere. 

# Coin Flipping Part Deux
I referenced coin flipping earlier for demonstrating the Sum Law. Now let's treat this problem more fully using
Bayes' Theorem. Let $p(\theta)$ be the probability of observing heads. $x$ is the number of heads we observe after
$n$ flips. We want to formulate a posterior probability for $\theta$ given some likelihood and prior. Likelihood
probabilities typically are strongly related to the structure of the problem. Our problem is determining the
probability of a discrete event occurring given so many trials. A quick gloss of the appendix of any statistics
textbook would lead us to choosing a **binomial distribution** for modeling this likelihood:

$$Pr(x | n, \theta) = {n \choose x} \theta^x (1 - \theta)^{n-x}$$

Now on to the prior. There are really two guiding principles for selecting a prior distribution: 1) domain knowledge
and 2) mathematical convenience. If we have some deep knowledge about the problem, then we could encode that information
into our posterior belief by way of the prior belief. For example, if I truly believed the coin was fair, I may
just set $p(\theta) = 0.5$ and call it a day. Of course this would mean that if the coin was unfair I may not be able
realize this in my posterior belief. Therefore, selecting a distribution is more appropriate. Which distribution
should I select? Again, maybe prior knowledge of the problem will help you here. Perhaps we think the fairness
of the coin is Gaussian distributed, or, for some odd reason we may believe it's Gamma distributed. In practice
though we want a prior distribution that will not make our life difficult. We Bayesians are lazy. Remember that
we will soon have to solve $\int p(a|b) p(b) db$, so let's choose a distribution that will make this easy for us.
Prior distributions that play well with their likelihood companion are **conjugate priors**. [Here](https://en.wikipedia.org/wiki/Conjugate_prior)
is a great Wikipedia page on conjugacy that includes a table of common conjugate priors. The prior
distribution conjugate to the Binomial distribution is the Beta distribution:

$$p(\theta | \alpha, \beta) = \frac{1}{\int^1_0 \theta^{\alpha - 1} (1- \theta)^{\beta-1}d\theta} \theta^{\alpha -1} (1-\theta)^{\beta-1}$$

The equation above may look messy, but we should just understand it as the continuous analog of the Binomial distribution.

With our likelihood and prior beliefs in hand, we can now formulate our posterior belief of $\theta$:

$$p(\theta| x, n, \alpha, \beta) = \frac{Pr(x | n, \theta) p(\theta | \alpha, \beta)}{\int Pr(x | n, \theta) p(\theta | \alpha, \beta) d\theta}$$

The algebra for reducing the above is unfortunately pretty hairy but because of our careful selection of a conjugate
prior, we know that the result will itself be a Beta distribution. It has the following form:

$$p(\theta| x, n, \alpha, \beta) = Beta(\alpha + x, \beta + n - x)$$

We're finished! We've modeled the probability of flipping heads as a Beta distribution incorporating the result of
trials, the number of trials, and our prior beliefs. You may wonder at this point what we should set $\alpha$ and $\beta$
to in this model. Again, this relies on prior beliefs. We can set both to 1 which will have the result of saying each
possible value of $\theta$ is equally likely. We may set $\alpha = 500$ and $\beta = 1000$ which would indicate that we're
pretty confident that the coin is fair.  We can even make the problem more complex and place priors on our priors,
 which are called **hyperpriors**. This process of selecting priors is called **prior elicitation** and is a
key problem in Bayesian machine learning. If we select good priors, the model will converge much faster than
traditionally methods. If we are egregiously wrong it will take much longer to reach convergence. We must
balance picking mathematically convenient priors with priors that reflect the problem fairly. In the case of coin
flipping, this is relatively easy. In machine learning this proves to be far more difficult.

# Reasoning With The Posterior
This may all seem like a lot of work. Why can't we just work with the likelihood directly and call it a day?
That is essentially what we do in most other machine learning methods that rely on **maximum likelihood estimation** (MLE)
for parameter fitting, but it does have one big limitation. The results of MLE methods are not probabilities, they
are point estimates. This unfortunately means we cannot quantify our uncertainty of the prediction rigorously 
which can be very problematic. At the end of the day, machine learning is about automated
decision making. We humans weigh our decisions by the certainty we have in the evidence. If you believe that
it will rain tomorrow because the weather man was right yesterday, your belief that it will rain the day after that
will inevitably be weighted by whether the weather will be right tomorrow. Our models should follow the same
principles to make the right decisions. Luckily, Bayesian methods make this very easy for us. A posterior belief
is much more powerful than a point estimate since it is represented as a probability distribution. Consider
the posterior belief from the previous section $p(\theta| x, n, \alpha, \beta)$. If we wanted to compute
a **95% credible interval** for this belief, meaning construct some interval where it is 95% likely for the parameter
$\theta$ to lie, we simply find bounds where the following is true

$$0.95 = Pr(\theta \in (a, b)| x, n, \alpha, \beta) = \int^a_b p(\theta| x, n, \alpha, \beta) d\theta$$

Similarly, we can **hypothesis test** our posterior belief by working with the same integral representation. We can
formulate the hypothesis that $\theta=0.5$, i.e. the coin is fair, as

$$p(\theta = 0.5| x, n, \alpha, \beta) = \int^{0.5}_{0.5} p(\theta = 0.5| x, n, \alpha, \beta) d\theta$$

The result of the hypothesis test above is more useful than its Frequentist counterpart since we receive a probability 
of the hypothesis being true rather than a mere indication to accept or reject.

Clearly the posterior belief has a central role in Bayesian machine learning, and when it takes form
of a known distribution like the Beta distribution, integrating it and using it for predictions is trivial.
Our world though is more complex than what the pages of a mathematical appendix can describe. We are often left working
with **intractable posteriors** when our likelihood and prior are not conjugate. For example, in Bayesian
logistic regression the Bernoulli likelihood and Gaussian prior lead us to a posterior we cannot express analytically.
We are not totally at a loss however. Instead, we can use methods of approximation. Often times we're not interested
in all the details of a posterior belief. Again considering Bayesian logistic regression, we really just need
the expectation of the posterior to perform inference. A common trick in this case is to use the **Laplace Approximation**
which superimposes a Gaussian over the intractable distribution and sets its mean to the mode of the intractable.
Other methods include **Expectation Propagation** and **Variational Inference** that are more common in complex
models like **Gaussian processes** and **Deep Bayes Nets**. We can also use traditional computation methods like
**Markov Chain Monte Carlo** for approximating posteriors though in practice those are too intense to include in a
training loop for machine learning.

# Known Unknowns
Bayesian machine learning is a large and growing field. Some areas of active research naturally follow some
problems we discussed above such as how to pick good priors and how to deal with tricky posteriors. Others areas
aim to Bayes-ify results from other fields. Simply google "Bayesian method-of-choice" and be inundated with the likes
of Bayesian LSTMs, Bayesian PCA, Bayesian decision trees etc. It has been extremely useful for statisticians and scientists
 to be able to quantify how much they do not know, and as machine learning continues to share the brunt
of decision-making, working with uncertainty will only become more important.
