---
date: 2021-06-02 02:01
title: "ML basics - Probability Part 2"
categories: DevCourse2 Probability MathJax
tags: DevCourse2 Probability MathJax
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

> **Deep Learning Model's Outcome is the Probability of the Variable X**

# Rules of Probability
## Product Rule
![lambda](/assets/images/product-rule.png){: .align-center}
\\[ p(s,y) = p(y|s)p(s) \\]
\\[ p(y,s) = p(s|y)p(y) \\]  

Eventhough we don’t know the intersection probability of R and O,  
if there are p(y=O|s=R) and p(s=R), we can get the prob of p(s=R, y=O)  
## Sum Rule
![lambda](/assets/images/sum-rule.png){: .align-center}  
\\[ p(X) = \sum_{Y}\ p(X,Y) \\]  



## Bayes Theorem
- posterior
- likelihood
- prior
- normalization
- marginal prob dist
- Conditional prob dist  

### Why Bayes Theorem is useful?
It's all about **dependency**.  
What I learned about intersection in school is that $$ P(A \cap B) = P(A)\ P(B) $$.  
This is only true when those two event A and B are **independent**.  
However, in reality this is not pleasible, this is why bayes theorm comes.  

\\[ P(H\mid E) = \frac{P(H)P(E\mid H)}{P(E)} \\]  

#### 3B1B example (Librarian or Farmer)
Prior state:  
The ratio is 20:1 (farmer:librarian).  
![lambda](/assets/images/20-farmers.png){: .align-center .img-40}  

**Is Steve more likely to be a librarian or a farmer?**  
“Steve is very shy and withdrawn, invariably helpful but with very little interest in people or in the world of reality. A meek and tidy soul, he has a need for order and structure, and a passion for detail.”  
Before given the information about meek and tidy, people decribe steve as more likely a farmer.  
However, after that, people view as a librarian.  
Accoding to Kahneman and Tversky, this is irrational.  
What's important is that almost **no one thinks to incorporate information about the ratio** of farmers to librarians into their judgements.  

- Sample 10 librarians and 200 farmers.
- Hear meek and tidy description.
- Your instinct: 40% of librarians and 10% of farmers
- Estimate: 4 librarians and 20 farmers  

The prob of random person who fits this description is a librarian is 4/24, or 16.7%.  
\\[ P(Librarian\ given\ Description) = \frac{4}{4+20} \thickapprox  16.7%\\]  

Even if you think a librarian is 4 times as likely as a farmer to fit this description(40% Vs. 10%), that's not enough to overcome the fact that there are way more farmers (prior state ratio).  
The key underlying Bayes' theorem is that new evidence does not completely determine your beliefs. It should **update** prior beliefs.  
![lambda](/assets/images/bayes-update.png){: .align-center .img-40}  

#### Heart of Bayes' Theorem
![lambda](/assets/images/heart-of-the-bayes.png){: .align-center .img-70}  

#### When to use Bayes' Theorem
![lambda](/assets/images/when-to-use-bayes.png){: .align-center .img-70}  

#### Prior and likelihood
![lambda](/assets/images/prior-likelihood.png){: .align-center .img-80}  

#### Posterior
![lambda](/assets/images/posterior.png){: .align-center}  

#### Normalization
In probability theory, a normalizing constant is a constant by which an everywhere non-negative function must be multiplied so the area under its graph is 1, e.g., to **make it a probability density function or a probability mass function**.  
베이즈 정리에서 $$P(E)$$ 가 normalizing constant 라고 보면된다.  


## Marginal Probaility istribution & Conditional Probaility Distribution
Marginal stands for the marginal of the table.  
Marginal probability doesn’t tell about the specific probability(i.e. each cell.)  

![lambda](/assets/images/marginal-conditional.png){: .align-center}  

# Functions of Random Variables
## Joint PDF
k차원의 확률변수 벡터 $$\boldsymbol x = (x_{1} , … , x_{k})$$ 가 주어졌을때,  
확률변수벡터 $$\boldsymbol y = (y_{1} , … , y_{k})$$ 를 정의한다.  
$$\boldsymbol y =\boldsymbol g(\boldsymbol x)$$ 로 나타낼 수 있다.  
ex. $$\boldsymbol y_{1} =\boldsymbol g_{1}(x_{1} , … , x_{k})$$  
만약 $$\boldsymbol y =\boldsymbol g(\boldsymbol x)$$ 가 일대일 변환인 경우,  
$$\boldsymbol y$$의 Joint PDF는  
\\[ P_{\boldsymbol y}\ (y_{1} , … , y_{k}) = P_{\boldsymbol x}\ (x_{1} , … , x_{k})\ |J| \\]  
$$J$$ is the matrix of all its first-order partial derivatives.*(wikipedia)*

### Jacobian Matrix
Jacobian Matrix: is the matrix of all first-order partial derivatives of a multiple variables vector-valued function. When the matrix is a square matrix, both the matrix and its determinantare referred to as the Jacobian in literature. The Jacobian is then the generalization of the gradient for vector-valued functions of several variables.  
> first-order partial derivatives: The first order derivative of a function represents the rate of change of one variable with respect to another variable. For example, in Physics we define the velocity of a body as the rate of change of the location of the body with respect to time. Here location is the dependent variable on the other hand time is the independent variable.  To find the velocity, we need to compute the first order derivative of the location. Similarly, we can this concept for computing rate of dependency of one variable over the other. <https://www.toppr.com/guides/fundamentals-of-business-mathematics-and-statistics/calculus/derivative-first-order/>  

Note that when m=1 the Jacobian is same as the gradient because it is a generalization of the gradient.  <https://math.stackexchange.com/questions/1519367/difference-between-gradient-and-jacobian>  

Jacobian Matrix가 Back propagation에 쓰이는 거 같은데 이건 내일 다시.  


## Inverse CDF Technique
# Expectations
## Variance
## Covariance
# Frequentist Vs. Bayesian
# Gaussian Distribution
## Normalization
## Expectation
## Variance
## Maximum Likelihood Solution
# Curve Fitting
# Bayesian Curve Fitting


# Rules of Probability

# Appendix
## MathJax
$$\approx$$:  
```
\approx
```  
$$\thickapprox$$:  
```
\thickapprox
```

## Mutually Exclusive Vs. Independent
**Mutually exclusive** events cannot happen at the same time. For example: when tossing a coin, the result can either be heads or tails but cannot be both.  
Events are **independent** if the occurrence of one event does not influence (and is not influenced by) the occurrence of the other(s). For example: when tossing two coins, the result of one flip does not affect the result of the other.  

**Mutually Exclusive**:  
$$\begin{align}
P(A \cap B) & = 0 \\
P(A \cup B) & = P(A) + P(B) \\
P(A|B) & = 0 \\
p(A|\neg B) & = \frac{P(A)}{1-P(B)} \\
\end{align}$$  

**Independent**:  
$$\begin{align}
P(A \cap B) & = P(A)\ P(B) \\
P(A \cup B) & = P(A) + P(B) - P(A)\ P(B) \\
P(A|B) & = P(A) \\
p(A|\neg B) & = P(A) \\
\end{align}$$  

## Gradient, Jacobian, Hessian
Gradient: Is a multi-variable generalization of the derivative. While a derivative can be defined on functions of a single variable, for scalar functions of several variables, the gradient takes its place. The gradient is a vector-valued function, as opposed to a derivative, which is scalar-valued.  
Jacobian Matrix: is the matrix of all first-order partial derivatives of a multiple variables vector-valued function. When the matrix is a square matrix, both the matrix and its determinantare referred to as the Jacobian in literature. The Jacobian is then the generalization of the gradient for vector-valued functions of several variables.  
Hessian Matrix: is a square matrix of second-order partial derivatives of a scalar-valued function, or scalar field. It describes the local curvature of a function of many variables. It is the Jacobian of the gradient of a scalar function of several variables.
## References
> Product Rule & Sum Rule: <https://www.youtube.com/watch?v=u7P9hg1dVDU>  
> MathJax Tex command: <https://www.onemathematicalcat.org/MathJaxDocumentation/MathJaxKorean/TeXSyntax_ko.html>

> mutually exclusive: <https://math.stackexchange.com/questions/941150/what-is-the-difference-between-independent-and-mutually-exclusive-events>  
> Probability Theory: <https://blog.naver.com/rokpilot/220603888519>  
> Gradient, Jacobian, Hessian: <https://www.quora.com/What-is-the-difference-between-the-Jacobian-Hessian-and-the-gradient-in-machine-learning>  
> Jacobian and gradient: <https://math.stackexchange.com/questions/1519367/difference-between-gradient-and-jacobian>  
