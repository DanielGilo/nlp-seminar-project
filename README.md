# nlp-seminar-project

The file astar_cf.py contains the algorithm itself.
The other two scripts, anytime_experiment.py and simulate_algorithm.py are used to conduct the reported experiments. 
In order to reproduce the reported results, one first has to run anytime_experiment.py, which generates raw results, and then simulate_algorithm.py, which simulates running the anytime algorithm for different allocated resources.
However, anytime_experiment.py assumes access to several files:
 - The black box classifier, that gets a text as input and return a binary classification in addition to its confidence level
 - The labeled training set
 - The classifier-labeled algorithm test set
 - Fine-tuned BERT for masked language modeling 
 - Fine-tuned GPT2 model for language modeling

Therefore, before running anytime_experiment.py, one has to train and fine tune the models as explained, and modify the first few lines of the main function to find the datasets and trained models.

