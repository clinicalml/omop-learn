# omop-learn

## What is omop-learn?

This library was developed in order to facilitate rapid prototyping in Python of predictive machine-learning models using longitudinal medical data from an OMOP CDM-standard database. omop-learn supports the easy definition of predictive clinical tasks, featurizations of OMOP data, and cohorts of relevance. We further provide methods using sparse tensor implementations to rapidly manipulate the collected features in the rawest form possible, allowing for dynamic transformations of the data.

Two machine-learning models are included with the library. First, a windowed linear model, which uses various backwards-facing windows to aggregate features over different timescales, then feeds these features into a regularized logistic regression model. This model was inspired by the work of [Razavian et. al. '15](https://people.csail.mit.edu/dsontag/papers/RazavianEtAl_BigData15.pdf), and despite its simplicity is often competitive with state-of-the-art algorithms. We also include SARD (Self-Attention with Reverse Distillation), a novel deep-learning algorithm that uses self-attention to allow medical events to contextualize themselves using other events in a patient's timeline. SARD also makes use of reverse distillation, a training technique we introduce that effectively initializes a deep model using a high-performing linear proxy, in this case the windowed linear model described above -- for the details of this method and the SARD architecture, please see our paper [Kodialam et al. '20](https://arxiv.org/abs/2007.05611).

## Documentation

For a more detailed summary of omop-learn's data collection pipeline, and for documentation of functions, please see the full [documentation](https://clinicalml.github.io/omop-learn/) for this repo, which also describes the process of creating one's own cohorts, predictive tasks, and features. 

## Dependencies

The following libraries are necessary to run omop-learn:

- numpy
- sqlalchemy
- pandas
- torch
- sklearn
- matplotlib
- ipywidgets
- IPython.display
- gensim.models
- scipy.sparse
- sparse 

Note that `sparse` is the PyData Sparse library, documented [here](https://sparse.pydata.org/en/stable/install.html)

## Running omop-learn

We provide two example notebooks, `End of Life Linear Model Example.ipynb` and `End of Life Deep Model Example` which both use an example task of predicting mortality over a six-month window for patients over the age of 70. The two notebooks run the windowed linear and deep SARD models respectively -- note that your machine must be able to access a GPU in order to run the deep models.

To run the models, first set up the file `config.py` with connection information for your Postgres server containing an OMOP CDM database. Then, simply run through the cells of the notebook in order. Further documentation of the exact steps taken to define a task, collect data, and run a predictive model are embedded within the notebooks. 


## Contributors and Acknowledgements

Omop-learn was written by Rohan Kodialam and Jake Marcus, with additional contributions by Rebecca Boiarsky, Ike Lage, and Shannon Hwang.

This package was developed as part of a collaboration with Independence Blue Cross and would not have been possible without the advice and support of Aaron Smith-McLallen, Ravi Chawla, Kyle Armstrong, Luogang Wei, and Jim Denyer.
