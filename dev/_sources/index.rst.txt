.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to stability-selection's documentation!
===============================================

This project contains an implementation of the stability selection algorithm.

Stability selection is a technique that aims to enhance and improve existing feature
selection algorithms. For a generic feature selection algorithm, we have a tuning
parameter :math:`\lambda \in \Lambda` that controls the amount of regularisation. Examples
of such algorithms are:

1. :math:`\ell_1`-penalized regression (penalization parameter :math:`\lambda`).
2. Orthogonal matching pursuit (number of steps in forward selection).
3. Boosting (:math:`\ell_1` penalty)

These structure learning algorithms have in common is a parameter :math:`\lambda \in \Lambda`
that controls the amount of regularisation. For every value of :math:`\lambda`, we obtain a structure
estimate :math:`S^\lambda = \{1, \ldots, p\}`, which indicates which variables to select. We are
interested to determine whether there exists a :math:`\lambda` such that :math:`S^\lambda` is identical to
:math:`S` with high probability, and how to achieve the right amount of regularisation.


Sability selection works as follows:

1. Define a candidate set of regularization parameters :math:`\Lambda` and a subsample number :math:`N`.
2. For each value :math:`\lambda \in \Lambda` do:

    a. For each :math:`i` in :math:`\{1, \ldots, N\}`, do:

        i. Generate a bootstrap sample of the original data :math:`X^{n \times p}` of size :math:`\frac{n}{2}`.
        ii. Run the selection algorithm (LASSO) on the bootstrap sample with regularization parameter :math:`\lambda`.

    b. Given the selection sets from each subsample, calculate the empirical selection probability for each model component:

       :math:`\hat{\Pi}^\lambda_k = \mathbb{P}[k \in \hat{S}^\lambda] = \frac{1}{N} \sum_{i = 1}^N \mathbb{I}_{\{k \in \hat{S}_i^\lambda\}}.`

    c. The selection probability for component :math:`k` is its probability of being selected by the algorithm.

3. Given the selection probabilities for each component and for each value of :math:`\lambda`, construct the
   stable set according to the following definition:

   :math:`\hat{S}^{\text{stable}} = \{k : \max_{\lambda \in \Lambda} \hat{\Pi}_k^\lambda \geq \pi_\text{thr}\}.`

   where :math:`\pi_\text{thr}` is a predefined threshold.

This algorithm identifies a set of “stable” variables that are selected with high probability.


    .. toctree::
       :maxdepth: 2
       
       api
       auto_examples/index
       ...

See the `README <https://github.com/thuijskens/stability-selection/blob/master/README.md>`_
for more information.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

