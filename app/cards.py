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
        'front-content':'Four strategies',
        'back-content':Markup('<ol><li>Reduce complexity of model</li>'
                              '<li>Add dropout layer(s)</li>'
                              '<li>Use more data for training/validation</li>'
                              '<li>Use data augmentation for training</li>'
                              '</ol>')
        },
        {
        'title':'PCA',
        'deck':"DS",
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
        'deck':"DS",
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
        'deck':"DS",
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
                              '<li> Used in multi-class classification problems where you want each output node to predict a number between 0 and 1</li>'
                              '</li>'
                              '</ol>')
        },
        ]