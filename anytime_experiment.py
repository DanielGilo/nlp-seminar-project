from astar_cf import AStar
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import scipy
import torch
import math
from transformers import pipeline
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
import editdistance as ed
import classifiers
import time
import pickle

# Load pre-trained model tokenizer (vocabulary)
#tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')







class Experiment:

    def __init__(self, experiment_name: str, to_save_in_directory_path: str, x, y, classifier, bert_mask, lm,
                 pos_scores_hist_path: str, confidence_level: float, sample_size: int, heuristic: str, heuristic_weight: float,
                 enable_word_addition: bool, use_antonyms: bool, use_bert: bool, enable_word_deletion: bool, branching_factor: int,
                 use_pos_analysis: bool, max_expensive_calls: int, pos_score_heuristic_weight: int,
                 bert_dicts, gpt2_dicts, tokenizer):
        self.experiment_name = experiment_name
        self.to_save_in_directory_path = to_save_in_directory_path
        self.x = x
        self.y = y
        self.classifier = classifier
        self.bert_mask = bert_mask
        self.lm = lm
        self.confidence_level = confidence_level
        self.sample_size = sample_size
        self.x_samples = self.x.loc[:self.sample_size - 1]
        self.heuristic = heuristic
        self.heuristic_weight = heuristic_weight
        self.enable_word_addition = enable_word_addition
        self.enable_word_deletion = enable_word_deletion
        self.bert_dicts = bert_dicts
        self.gpt2_dicts = gpt2_dicts
        self.branching_factor = branching_factor
        self.use_pos_analysis = use_pos_analysis
        self.max_expensive_calls = max_expensive_calls
        self.pos_score_heuristic_weight = pos_score_heuristic_weight
        self.tokenizer = tokenizer
        self.use_antonyms = use_antonyms
        self.use_bert = use_bert

        with open(pos_scores_hist_path, 'rb') as f:
            self.pos_scores_hist = pickle.load(f)

        self.sentences_results = []
        self.time_dicts = []

    def gpt2_score(self,sentence):
        tokenize_input = self.tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss = self.lm(tensor_input, lm_labels=tensor_input)
        return math.exp(loss)

    def run(self):
        for i in range(self.sample_size):
            self.sentences_results.append(self.run_sentence(i))
            # with open(self.to_save_in_directory_path + "gpt2_dict_{}.pkl".format(self.heuristic_weight), 'wb') as f:
            #     pickle.dump(self.gpt2_dict, f)
            # with open(self.to_save_in_directory_path + "bert_dict_{}.pkl".format(self.heuristic_weight), 'wb') as f:
            #     pickle.dump(self.bert_dict, f)



    def run_sentence(self, sentence_index):
        results = {"sentence_index": -1, "success": False, "confidence_level": -1, "distance": -1, "running_time": -1, "expensive_calls": -1,
                   "language_model_ratio": -1,"original_predicted_label": -1, "original_sentence": "",
                   "counterfactual": ""}
        sentence = self.x_samples[sentence_index].replace(' , ', ', ').replace(' .', '.').lower()
        bert_dict = self.bert_dicts[sentence_index]
        gpt2_dict = self.gpt2_dicts[sentence_index]
        results["original_sentence"] = sentence
        results["original_predicted_label"], _ = self.classifier.predict(sentence)
        results["sentence_index"] = sentence_index

        Astar = AStar(pipeline=self.bert_mask, lm=self.lm,
                      black_box=self.classifier, original_label=results["original_predicted_label"], pos_scores_histogram= self.pos_scores_hist,
                      distance_metric="edit", heuristic=self.heuristic,
                      use_antonyms=self.use_antonyms, confidence_threshold=self.confidence_level,
                      max_gpt_ratio=1.5, max_expensive_calls= self.max_expensive_calls, branching_factor=self.branching_factor,
                      heuristic_weight=self.heuristic_weight,
                        enable_word_addition=self.enable_word_addition,enable_word_deletion=self.enable_word_deletion,
                      bert_dict=bert_dict, gpt2_dict=gpt2_dict, use_pos_analysis=self.use_pos_analysis,
                      pos_score_heuristic_weight=self.pos_score_heuristic_weight, tokenizer=self.tokenizer, use_bert=self.use_bert)
        start = time.time()
        path, results["expensive_calls"], time_dict = Astar.astar(sentence)
        self.time_dicts.append(time_dict)
        end = time.time()
        results["running_time"] = (end - start)
        if isinstance(path, type(None)):
            results["success"] = False
            return results

        results["success"] = True
        path = list(path)
        cf = path[-1].replace(' , ', ', ').replace(' .', '.').replace('.','').lower()
        _, results["confidence_level"] = self.classifier.predict(cf)
        results["distance"] = ed.eval(sentence.replace('.','').split(" "), cf.split(" "))
        results["language_model_ratio"] = self.gpt2_score(cf)/self.gpt2_score(sentence)
        results["counterfactual"] = cf.replace(' , ', ', ').replace(' .', '.')

        self.bert_dicts[sentence_index] = Astar.bert_dict
        self.gpt2_dicts[sentence_index] = Astar.gpt2_dict
        return results

    def save_results(self):
        full_path_res = self.to_save_in_directory_path + self.experiment_name + "_results.pkl"
        full_path_times = self.to_save_in_directory_path + self.experiment_name + "_times.pkl"
        with open(full_path_res, 'wb') as f:
            pickle.dump(self.sentences_results, f)
        with open(full_path_times, 'wb') as f2:
            pickle.dump(self.time_dicts, f2)


def get_baseline_cf(x, y, sentence_label, original_sentence):
    min_dist = np.inf
    cf = None
    for i in range(len(x)):
        new_sentence = x.loc[i].replace(' , ', ', ').replace(' .', '.')
        d = ed.eval(original_sentence.split(" "), new_sentence.split(" "))
        if (0 < d < min_dist) and (y.loc[i] != sentence_label):
            cf = new_sentence
            min_dist = d
    return cf, min_dist


if __name__ == '__main__':
    ds_name = "Yelp"
    starting_index = 0
    hw_list = [1.0, 0.75, 0.5, 0.25, 0.0]
    df_train = pd.read_csv("datasets/{}/train.csv".format(ds_name+"_long"), header=0)
    x_train, y_train = df_train["text"], df_train["label"]
    x_alg_test, predicted_y_alg_test = pd.read_csv("datasets/{}/predicted_alg_test.csv".format(ds_name+"_long"), header=0)["text"],\
                            pd.read_csv("datasets/{}/predicted_alg_test.csv".format(ds_name+"_long"), header=0)["label"]
    x_alg_test, predicted_y_alg_test = x_alg_test.loc[starting_index:].reset_index(drop=True), predicted_y_alg_test[starting_index:].reset_index(drop=True)
    classifier = classifiers.CustomClassifier("models/{}/".format(ds_name))
    bert_mask = pipeline("fill-mask", model="fine_tuning/{}/bert/".format(ds_name))
    lm = OpenAIGPTLMHeadModel.from_pretrained('fine_tuning/{}/gpt2/'.format(ds_name))
    tokenizer = OpenAIGPTTokenizer.from_pretrained('fine_tuning/{}/gpt2/'.format(ds_name))
    lm.eval()
    budget = 200
    sample_size = 3
    bert_dicts = []
    gpt2_dicts = []
    for i in range(sample_size):
        bert_dicts.append({})
        gpt2_dicts.append({})
    pos_scores_hist_path = "objects/stanford_140/score_hist.pkl" #doesn't really matter since pos_score_hw = 0.0
    for hw in hw_list:
        print("****** hw = {}".format(hw))
        experiment = Experiment(experiment_name="hw={}_start_{}_end_{}".format(hw, starting_index,starting_index + sample_size - 1),
                                to_save_in_directory_path="results/Paper/{}/specific+agnostic/".format(ds_name),
                                x=x_alg_test, y=predicted_y_alg_test, classifier=classifier, bert_mask=bert_mask,  lm=lm,
                                pos_scores_hist_path=pos_scores_hist_path, bert_dicts=bert_dicts, gpt2_dicts=gpt2_dicts,
                                confidence_level=0.95, sample_size=sample_size, heuristic="bb confidence",
                                heuristic_weight=hw, enable_word_addition=False, enable_word_deletion=False, branching_factor=np.inf,
                                use_pos_analysis=False, max_expensive_calls=budget, pos_score_heuristic_weight=0.0,
                                tokenizer=tokenizer, use_antonyms=True, use_bert=True)
        experiment.run()
        experiment.save_results()
        bert_dicts = experiment.bert_dicts
        gpt2_dicts = experiment.gpt2_dicts


        for k in experiment.time_dicts[0].keys():
            _sum = sum(d[k] for d in experiment.time_dicts)
            print("{} total: {}".format(k, _sum))

        for d in experiment.sentences_results:
            print(d)

    # with open(directory+"gpt2_dict", 'wb') as f:
    #     pickle.dump(experiment.gpt2_dict, f)
    #
    # with open(directory+"bert_dict", 'wb') as f:
    #     pickle.dump(experiment.bert_dict, f)






