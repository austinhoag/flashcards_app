from flask import Markup

card_data = [
        {
        'title':'How to tackle underfitting',
        'deck':"DS",
        'front-content':'Four strategies',
        'back-content':Markup('<ol><li>Increase complexity of model</li>'
                              '<li>Try different types of layers/number of nodes per layer</li>'
                              '<li>Increase number of input features (if possible)</li>'
                              '<li>Decrease fraction of dropout nodes (if possible)</li>'
                              '</ol>')
        },
        {
        'title':'How to tackle overfitting',
        'deck':"DS",
        'front-content':'List at least four strategies',
        'back-content':Markup('<ol><li>Reduce complexity of model</li>'
                              '<li>Add dropout layer(s)/regularization</li>'
                              '<li>cross-validated evaluation</li>'
                              '<li>Use more data for training/validation</li>'
                              '<li>Use data augmentation for training</li>'
                              '<li>Ensemble learning</li>'
                              '</ol>')
        },
        {
        'title':'Basic parts of Deep Learning',
        'deck':"DS",
        'front-content':'Six parts',
        'back-content':Markup('<ol><li>Data</li>'
                              '<li>Loss function (e.g. cross entropy)</li>'
                              '<li>Optimization algorithm (e.g. Adam)</li>'
                              '<li>Network architecture (e.g. Dense layers)</li>'
                              '<li>Test data</li>'
                              '<li>Evaluation metric (e.g. Accuracy)</li>'
                              '</ol>')
        },
        {
        'title':'PCA',
        'deck':"Stats",
        'front-content':Markup('<ol><li>What it does?</li>'
                              '<li>When to use it?</li>'
                              '<li>How to calculate PCs?</li>'
                              '</ol>'),
        'back-content':Markup('<ol><li>PCA finds the principle components (PCs) of a dataset. These are linear combinations of your input features that maximize the variance in the data</li>'
                              '<li> Used for dimensionality reduction or to orthogonalize data</li>'
                              '<li>Standardize data (unit std and 0 mean), calculate covariance matrix, find the eigenvectors & eigenvalues of cov matrix</li>'
                              '</ol>')
        },
        {
        'title':'Common types of layers in a NN',
        'deck':"DS",
        'front-content':Markup('<ol><li>Names</li>'
                              '<li>Definitions</li>'
                              '<li>When to use</li>'
                              '</ol>'),
        'back-content':Markup('<ol><li>Dense (aka fully connected) layers<ul>'
        							'<li>Connected to all adjacent input and output nodes</li>' 
        							'<li>Often used as final layer in classification tasks</li>'
        					  '</ul></li>'
        					  '<li>Convolutional layers<ul>'
        							'<li>Convolves the inputs with a number of filters</li>' 
        							'<li>Used in image recognition tasks/signal processing</li>'
        					  '</ul></li>'
        					  '<li>Dropout layers<ul>'
        							'<li>Not a real layer but acts on layers</li>' 
        							'<li>Ignores a fraction of nodes in a layer during both forward and back propagation </li>'
        							'<li>Can help to not overfit the training data -- helps with model generalization </li>'
        					  '</ul></li>'
                              '</ol>')
        },
        {
        'title':'Binomial Distribution',
        'deck':"Stats",
        'front-content':Markup('<ol><li>Definition (in words)</li>'
                              '<li>Mathematical formula</li>'
                              '<li>assumptions</li>'
                              '<li>mean/variance</li>'
                              '</ol>'),
        'back-content':Markup('<ol><li>Discrete probability distribution of k successes in n trials, with p probability of success in each trial</li>'
                              '<li>  $$P(k;n,p) = {n\choose k} p^k (1-p)^{n-k}$$</li>'
                              '<li>Assumes each trial is independent and has the same probability</li>'
                              '<li>$$\mu = np \quad\quad \sigma = np(1-p)$$</li>'
                              '</ol>')
        },
        {
        'title':'Kullback Leibler (KL) Divergence',
        'deck':"Stats",
        'front-content':Markup('<ol><li>Mathematical formula</li>'
                              '<li>Interpretation in Machine Learning</li>'
                              '<li>Also known as what?</li>'
                              '</ol>'),
        'back-content':Markup('<ol><li>$$D_{KL} (P||Q) = \sum_{x \epsilon X}{ln({P(x) \over Q(x)})} $$</li>'
                              '<li> Interpretation: information gain if model P is used instead of Q.  </li>'
                              '<li> Also called: relative entropy</li>'
                              '</ol>')
        },
        {
        'title':'Batch normalization',
        'deck':"DS",
        'front-content':Markup('<ol><li>What is it?</li>'
                              '<li>Why is it used?</li>'
                              '<li>Mathematical formula</li>'
                              '</ol>'),
        'back-content':Markup('<ol><li>A method for normalizing outputs from an activation function of a NN layer</li>'
                              '<li> It is used to combat exploding gradients problem</li>'
                              '<li> Normalize output then add multiplicative and additive factor:</li>'
                              '$$ z = {(x-\mu) \over \sigma} \quad\quad \mathrm{Norm} = z*g+b $$</li>'
                              '</ol>')
        },
        {
        'title':'Softmax activation function',
        'deck':"DS",
        'front-content':Markup('<ol><li>Mathematical formula</li>'
                              '<li>What layers in a NN typically use it</li>'
                              '<li>Why is it used?</li>'
                              '</ol>'),
        'back-content':Markup('<ol><li>$$ \sigma(x) = {e^{x_i} \over \sum_{j=1}^{K} e^{x_j}} $$</li>'
                              '<li> Typically used in dense (aka fully connected layers)</li>'
                              '<li> Used in multi-class classification problems where you want each output node to predict a number between 0 and 1 (probability)</li>'
                              '</li>'
                              '</ol>')
        },
        {
        'title':"Harris' distributional hypothesis",
        'deck':"DS",
        'front-content':'What is the hypothesis?',
        'back-content':"Words that occur in similar contexts tend to have similar meanings"
        },
        {
        'title':"Ridge Regression",
        'deck':"Stats",
        'front-content':Markup('<ol><li>What is it?</li>'
                              '<li>What kind of models can it be used for?</li>'
                              '<li>Mathematical formula (the bias term)</li>'
                              '<li>Why is it used?</li>'
                              '</ol>'),
        'back-content':Markup('<ol><li>A regularization technique where a bias is added to minimize variance. </li>'
                              '<li>Linear regression, logistic regression</li>'
                              '<li>For linear regression: add the following term to the cost function: $$\lambda m^2$$ where lambda is a hyperparameter determined via cross validation and m is the slope. In general, ridge regression penalty contains all parameters except the y-intercept.</li>'
                              '<li>Used to desensitize model to training data bc it penalizes large slopes (think "y/x"). It can also allow you to fit models with more parameters than data points. See: <a href="https://www.youtube.com/watch?v=Q81RR3yKn30">YouTube</a></li>'
                              '</ol>'),
        },
        {
        'title':"Precision, Recall and F1 Score",
        'deck':"DS",
        'front-content':"Definitions of all three in terms of TP, FP, TN, FN. Also, why is F1 defined that way?",
        'back-content':Markup('<ol><li>Precision: $$ {TP \over {TP + FP} }$$ </li>'
                              '<li>Recall: $$ {TP \over {TP + FN} }$$ </li>'
                              '<li>F1: Harmonic mean of precision and recall: $$ 2*{P*R \over {P + R} }$$.</li>'
                              '<li>F1 is harmonic mean of the two to punish extreme values of either P or R.</li>'
                              '</ol>'),
        },
        {
        'title':"Naive Bayes",
        'deck':"DS",
        'front-content':Markup("<ul><li>Why is it called 'naive?'</li>"
                               "<li> Mathematical formula for posterior odds </li>"
                               "<li> What model distribution is commonly used for text classification? </li>"
                               "</ul> </li>"
                               ),
        'back-content':Markup('<ul><li>Naive because you have to assume the model distribution -- e.g. that data are drawn from Gaussian distributions </li>'
                              '<li>Let L1 = label 1, L2 = label 2, '
                              '$$ {P(L1|\mathrm{features}) \over P(L2|\mathrm{features})} = {P(\mathrm{features}|L1)P(L1) \over P(\mathrm{features}|L2)P(L2)}$$ </li>'
                              '<li>Multinomial distribution is often used for text classification because '
                              'you have many words with difference frequencies (counts) and you want to classify documents based on those counts</li>'

                              '</ul>'),
        },
        {
        'title':'Type I and II errors',
        'deck':"Stats",
        'front-content':"What are they?",
        'back-content':Markup('<ul><li>Type I error = False positive</li>'
                              '<li>Type II error = False negative</li>'
                              '</ul>')
        },

        {
        'title':'A/B testing (multiple tests case)',
        'deck':"Stats",
        'front-content':Markup('Sample question: <ul><li>10 tests are running with different landing page</li>'
                               '<li>1 case won and the p-value is just under 0.05 </li>'
                               '<li>Should you make the change?</li>'
                               '</ul'),
        'back-content':Markup('No because: $$P(\mathrm{no\,false\,positive}) = (0.95)^{\mathrm{N_{tests}}} = 0.95^{10} = 0.6$$  '
                'so $$ P(\mathrm{false\,positive}) = 1-0.6 = 0.4$$'
                'Instead some methods are to use the <b>Bonferroni</b> correction (conservative, less power) which is: $$p < {0.05 \over \mathrm{N_{tests}}} = {0.05 \over 10} = 0.005$$'
                'There is also the less conservative <b>Benjamini-Hochberg</b> method (less conservative, more power). The method is on another flashcard'
                )
        },
        {
        'title':'A/B testing (single test case)',
        'deck':"Stats",
        'front-content':Markup('<ul><li>What is it in its simplest form?</li>'
                               '<li>Example</li>'
                               '<li>How to determine sample size?</li>'
                               '<li>Rule of thumb for determining sample size?</li>'
                               '</ul'),
        'back-content':Markup('<ul><li>Test treatment group B against control group A</li>'
                              '<li>Tech company tests new feature against existing feature on subset of users </li>'
                              '<li>To determine sample size you need: <ol><li>Type II error or power</li><li> significance</li><li> Minimum detectable effect</li></ol></li>'
                              '<li>$$\mathrm{Sample size} \simeq {16 \sigma^2 \over \delta^2} $$ where $$\sigma^2$$ is sample variance and $$\delta$$ is minimum difference you want to see between groups.  </li>'
                              '</ul>')
        },
        {
        'title':'Benjamini-Hochberg method',
        'deck':"Stats",
        'front-content':'What is the method used for and what are the steps for performing it?',
        'back-content':Markup('A less conservative method (i.e. retains more power) method than Bonferroni for adjusting p-values in multiple comparisons'
                '<ol>'
                '<li> Sort p-values </li>'
                '<li> Assign ranks to them </li>'
                '<li> The largest FDR adjusted p-value is the same as the largest p-value </li>'
                '<li> The next largest FDR adjusted p-value is </br>min(previously adjusted p-value, current p-value * (# of p-values / current p-value rank)) </li>'
                '</ol>This is a REALLY helpful explanation: <a href="https://www.youtube.com/watch?v=K8LQSvtjcEo">False Discovery Rates, FDR, clearly explained (StatQuest)</a>'
                )
        },
        {
        'title':'AdaBoost Algorithm',
        'deck':"DS",
        'front-content':Markup('<ul><li>Why is it used?</li><li>How to implement?</li> '),
        'back-content':Markup('Adaboost helps you combine multiple weak classifiers (e.g. single split decision trees) into a single strong classifier'
                '<ol>'
                '<li> Assign every observation an initial weight value: $$w_i = {1 \over n}$$ </li>'
                '<li> Train a weak model (most often a decision tree) </li>'
                '<li> For each observation: <ul><li>If predicted incorrectly, weight is increased</li><li>If predicted correctly, weight is decreased </li></ul></li>'
                '<li> Train a new weak model where observations with greater weights are given more priority </li>'
                '<li> Repeat steps 3 and 4 until observations perfectly predicted or a preset number of trees are trained. </li>'
                '</ol>This is a REALLY helpful explanation: <a href="https://www.youtube.com/watch?v=LsK-xG1cLYA">AdaBoost, Clearly Explained (StatQuest)</a>'
                )
        },
        {
        'title':'Gini impurity',
        'deck':"DS",
        'front-content':Markup('<ul><li>What is it?</li><li>How to calculate?</li> '),
        'back-content':Markup('It is a measure for determining the splits in a decision tree. Impurity refers to the fact that trees do not perfectly split the data into two classes.'
                '<ol>'
                '<li> Calculate the impurity for each leaf node: $$ 1 - P(\mathrm{yes})^2 - P(\mathrm{no})^2$$ </li>'
                '<li> Calculate the weighted average of each impurity based on number of observations in each leaf node </li>'
                '</ol>This is a REALLY helpful explanation: <a href="youtube.com/watch?v=7VeUPuFGJHk&t=430s">Decision Trees (StatQuest)</a>'
                )
        },
        {
        'title':'Regression trees',
        'deck':"DS",
        'front-content':Markup('<ul><li>What are they?</li><li>How are they different from decision trees?</li> '),
        'back-content':Markup('They are an alternative to linear regression that allows you to fit non-linear relationships in the data.'
                '<ul>'
                '<li> In regression trees, the nodes are numerical values, instead of True or False like in decision trees  </li>'
                '<li> To build one, split the data by picking thresholds that minimize the sum of the squared residuals </li>'
                '</ul>This is a REALLY helpful explanation: <a href="https://www.youtube.com/watch?v=g9c66TUylZ4">Regression Trees (StatQuest)</a>'
                )
        },
        {
        'title':'AIC',
        'deck':"Stats",
        'front-content':Markup('<ul><li>What is it and what is it used for?</li><li>How to calculate?</li><li>Assumptions</li> '),
        'back-content':Markup('It is a relative statistical estimator used for model comparison'
                '<ul>'
                '<li> Often used when you do not want to do traditional train/val/test split - e.g. in time series where recent data is most valuable </li>'
                '<li> Can think of it as evaluating a model\'s fit to the data plus a pentality for model complexity</li>'
                '<li> $$ \mathrm{AIC} = -2\ln{L} + 2k, $$ where L is likelihood (often just residual sum of squares) and k is number of params.</li>'
                '<li> Assumes infinite sample size. For smaller sample sizes use <a href="https://www.youtube.com/watch?v=7XAHjm6Vy5k&t=1045s">AICc</a></li>'
                '</ul>Helpful explanation: <a href="https://towardsdatascience.com/introduction-to-aic-akaike-information-criterion-9c9ba1c96ced">Towards Data Science</a>'
                )
        },
        {
        'title':'R squared (coefficient of determination)',
        'deck':"Stats",
        'front-content':'What is meant by R squared? Definition (in words)? Mathematical definition?',
        'back-content':Markup('<ol><li>It is a statistical measure of how close the data are to a fitted regression line. </li>'
                              '<li>It is the percentage of the response variable variation that is explained by a linear model. </li>'
                              '<li>$$R^2 = 1 - {\sum_i{(y_i - \hat{y_i})^2} \over \sum_i{(y_i - \overline{y})^2} },$$ where y_i is data, y_i hat is predicted, y bar is mean data </li>'
                              '</ol>')
        },
        {
        'title':'Cross entropy',
        'deck':'DS',
        'front-content':'Defintion (words and math), when is it used? Why is it preferred over sum of squared residuals?',
        'back-content':Markup('<ol><li>It is a loss function used for multi-class classification problems when your class outputs are probabilities. It generalizes log loss. </li>'
                              '<li>$$ S_c = - \sum_{c=1}^{M}{\mathrm{Observed}_c \\times \log{(\mathrm{Predicted}_c})}, $$ where c runs over the different classes (of where there are a total of M) </li>'
                              '<li>The log function has a steeper shape near 0 so better derivative between 0 and 1 for back prop than sum of squared residuals.'
                              'Helpful explanation: <a href="https://www.youtube.com/watch?v=6ArSys5qHAU">Cross Entropy (StatQuest)</a>')
        },
        {
        'title':'Ensemble methods (bagging, boosting, and stacking)',
        'deck':'DS',
        'front-content':'What are they, and when are they used? How do they differ from each other?',
        'back-content':Markup('<ol><li>Bagging stands for "bootstrap aggregation" and involves resampling the data and combines multiple (often) homogeneous classifiers learned in parallel following some averaging process. </li>'
                              '<li> Boosting does not resample the data and combines multiple (often) homogeneous classifiers learned <i>sequentially</i> following some known stragety. </li>'
                              '<li> Stacking does not resample the data and combines mutliple heterogeneous classifiers by training a meta-model to output a prediction based on the different weak models predictions </li>'
                              '<li> Bagging can reduce variance but not bias, boosting and stacking can reduce both. </li>')
        }
]