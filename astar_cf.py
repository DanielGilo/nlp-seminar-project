
#taken from: https://github.com/jrialland/python-astar/blob/master/src/astar/__init__.py

from abc import ABCMeta, abstractmethod
from heapq import heappush, heappop
import scipy
import random
from nltk.corpus import wordnet
import editdistance as ed
import numpy as np
import torch
import math
import classifiers
import time


from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
import spacy

#model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
#model = OpenAIGPTLMHeadModel.from_pretrained('fine_tuning/SST/gpt2/')
#model.eval()
# Load pre-trained model tokenizer (vocabulary)
# tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
#tokenizer = OpenAIGPTTokenizer.from_pretrained('fine_tuning/{}/gpt2/'.format("SST"))


#pos_tokenizer = spacy.load(r'/Users/danielgilo/opt/miniconda3/lib/python3.7/site-packages/en_core_web_sm/en_core_web_sm-2.3.1')
pos_tokenizer = {}


Infinite = float('inf')







class AStar:
    __metaclass__ = ABCMeta
    __slots__ = ('encoder', 'training_set_encodings', 'pipeline', 'original_label', 'black_box', 'distance_metric', 'heuristic', 'use_antonyms', 'use_bert',
                 'confidence_threshold', 'max_gpt_ratio', 'original_gpt_score',  'max_expensive_calls', 'pos_scores_histogram', 'branching_factor', 'use_pos_analysis',
                 'max_heuristic_score', 'max_distance', 'heuristic_weight', 'times_dict', 'enable_word_addition', 'enable_word_deletion', 'bert_dict', 'gpt2_dict',
                 'expensive_calls_counter', 'pos_score_heuristic_weight', 'lm', 'tokenizer', 'max_words_to_modify')


    def __init__(self, pipeline, lm, original_label, black_box: classifiers.Classifier, pos_scores_histogram,
                 bert_dict, gpt2_dict, tokenizer, distance_metric = "edit", heuristic = 'bb confidence', use_antonyms = True,
                 confidence_threshold = 0.5, max_gpt_ratio = np.inf, max_expensive_calls = 500, branching_factor = 10,
                 heuristic_weight = 0.5, enable_word_addition=False, enable_word_deletion=False,
                 use_pos_analysis = True, pos_score_heuristic_weight = 0, use_bert=True, max_words_to_modify=30):

        self.distance_metric = distance_metric
        self.pipeline = pipeline
        self.lm = lm
        self.original_label = original_label
        self.black_box = black_box
        self.heuristic = heuristic
        self.use_antonyms = use_antonyms
        self.confidence_threshold = confidence_threshold
        self.max_gpt_ratio = max_gpt_ratio
        self.max_expensive_calls = max_expensive_calls
        self.branching_factor = branching_factor
        self.heuristic_weight = heuristic_weight
        self.enable_word_addition = enable_word_addition
        self.enable_word_deletion = enable_word_deletion
        self.bert_dict = bert_dict
        self.gpt2_dict = gpt2_dict
        self.pos_scores_histogram = pos_scores_histogram
        self.use_pos_analysis = use_pos_analysis
        self.expensive_calls_counter = 0
        self.pos_score_heuristic_weight = pos_score_heuristic_weight
        self.tokenizer = tokenizer
        self.use_bert = use_bert
        self.max_words_to_modify = max_words_to_modify



        self.times_dict = {"pop time": 0, "pops": 0, "distance calc time": 0, "distance calc calls": 0, "neighbors calc time": 0,
                           "neighbors calc calls": 0, "gpt calc time": 0, "gpt calc calls": 0, "bert calc time": 0,
                           "bert calc calls": 0, "antonyms calc time": 0, "antonyms calc calls": 0}


    class SearchNode:
        __slots__ = ('data', 'gscore', 'fscore',
                     'closed', 'came_from', 'out_openset')

        def __init__(self, data, gscore=Infinite, fscore=Infinite):
            self.data = data
            self.gscore = gscore
            self.fscore = fscore
            self.closed = False
            self.out_openset = True
            self.came_from = None

        def __lt__(self, b):
            return self.fscore < b.fscore

    class SearchNodeDict(dict):

        def __missing__(self, k):
            v = AStar.SearchNode(k)
            self.__setitem__(k, v)
            return v

    def to_modify(self):
        prob = min(self.max_words_to_modify/self.max_distance, 1.0)
        x = random.random()
        if x < prob:
            return True
        return False

    def heuristic_cost_estimate(self, current_node):
        """Computes the estimated (rough) distance between a node and the goal, this method must be implemented in a subclass. The second parameter is always the goal."""
        current = current_node.data
        h_score = 0
        if self.heuristic != 'bb confidence':
            raise ValueError
        label, score = self.black_box.predict(current)
        if label == self.original_label:
            h_score = score - (1-self.confidence_threshold)
        else:
            if score < self.confidence_threshold: #shouldn't reach here if conf = 0.5
                h_score = self.confidence_threshold - score
            else: #reached goal
                h_score = 0

        if current_node.came_from is None:
            pos_score = 0
        else:
            # pos_score = self.get_pos_score(current_node)
            pos_score = 0.0
        h_score = (1-self.pos_score_heuristic_weight)*h_score + (self.pos_score_heuristic_weight*pos_score)
        return (1/self.confidence_threshold)*h_score

    def get_pos_score(self, current_node):
        cur_sentence = pos_tokenizer(str(current_node.data))
        prev_sentence = pos_tokenizer(str(current_node.came_from.data))
        for i in range(len(prev_sentence)):
            if prev_sentence[i].text != cur_sentence[i].text:
                pos = prev_sentence[i].pos_
                if (pos in self.pos_scores_histogram.keys()) and (self.pos_scores_histogram[pos] > 0):
                    return self.pos_scores_histogram[pos]
                else:
                    return self.pos_scores_histogram["X"]

        return self.pos_scores_histogram["X"] #default if for some reason no changes between node and father

    def gpt2_score(self, sentence):
        #self.expensive_calls_counter += 1
        try:
            if sentence in self.gpt2_dict:
                return self.gpt2_dict[sentence]
            self.expensive_calls_counter += 1
            gpt_time_start = time.time()
            tokenize_input = self.tokenizer.tokenize(sentence)
            tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)])
            loss = self.lm(tensor_input, lm_labels=tensor_input)
            score = math.exp(loss)
            gpt_time_end = time.time()
            self.times_dict["gpt calc time"] += gpt_time_end - gpt_time_start
            self.times_dict["gpt calc calls"] += 1
            self.gpt2_dict[sentence] = score
            return score
        except: #edge cases fail, probably empty sentences due to deleted words
            print("gpt2 threw exception for this sentence {}".format(sentence))
            return np.inf

    def distance_between(self, n1, n2):
        """Gives the real distance between two adjacent nodes n1 and n2 (i.e n2 belongs to the list of n1's neighbors).
           n2 is guaranteed to belong to the list returned by the call to neighbors(n1).
           This method must be implemented in a subclass."""
        dist_start_time = time.time()
        self.times_dict["distance calc time"]

        if self.distance_metric == 'edit':
            distance = ed.eval(n1.split(" "), n2.split(" "))
        else:
            # v1 = self.encoder.encode([n1])
            # v2 = self.encoder.encode([n2])
            # distance = scipy.spatial.distance.cdist(v1,v2, self.distance_metric)[0][0]
            raise ValueError

        dist_end_time = time.time()
        self.times_dict["distance calc time"] += dist_end_time - dist_start_time
        self.times_dict["distance calc calls"] += 1

        return distance/self.max_distance

    def is_candidate_valid(self, curr_s, candidate):
        candidate_score = self.gpt2_score(candidate)
        if (curr_s == candidate) or ((candidate_score / self.original_gpt_score) > self.max_gpt_ratio):
            return False
        return True

    def get_valid_deleted_words_candidates(self, s, s_arr, pos_scores):
        candidates = []
        for i in range(len(s_arr)):
            s_arr_copy = s_arr.copy()
            del s_arr_copy[i]
            candidate = ' '.join(s_arr_copy)
            if self.is_candidate_valid(s, candidate):
                candidates.append((candidate, pos_scores[i]))

        return candidates

    def get_bert_suggestions(self, s_masked):
        # self.expensive_calls_counter += 1
        if s_masked in self.bert_dict:
            return self.bert_dict[s_masked]
        else:
            self.expensive_calls_counter += 1
            bert_time_start = time.time()
            s = self.pipeline(s_masked)
            bert_time_end = time.time()
            self.times_dict["bert calc time"] += bert_time_end - bert_time_start
            self.times_dict["bert calc calls"] += 1
            self.bert_dict[s_masked] = s
            return s


    def get_valid_bert_candidates(self, s, s_arr, pos_scores):
        nlp = self.pipeline
        bert_candidates = []

        for i in range(len(s_arr)):
            if not self.to_modify():
                continue
            s_arr_copy = s_arr.copy()
            s_arr_copy[i] = nlp.tokenizer.mask_token
            new_s = ' '.join(s_arr_copy)
            filled_new_s = self.get_bert_suggestions(new_s)
            for j in range(5):
                if filled_new_s[j]['score'] < (filled_new_s[0]['score']*(1-0.5)):
                    break

                candidate = filled_new_s[j]['sequence'].replace('<s> ', '').replace('<s>', '').replace('</s>', '').replace('[CLS] ','').replace(' [SEP]', '')
                if self.is_candidate_valid(s, candidate):
                    already_in_candidates = False
                    for c in bert_candidates:
                        if c[0] == candidate:
                            already_in_candidates = True
                            break
                    if not already_in_candidates:
                        candidate_score = pos_scores[i]
                        bert_candidates.append((candidate, candidate_score))

        if self.enable_word_addition:
            for i in range(len(s_arr)):
                if not self.to_modify():
                    continue
                s_arr_copy = s_arr.copy()
                s_arr_copy.insert(i, nlp.tokenizer.mask_token)
                new_s = ' '.join(s_arr_copy)
                filled_new_s = nlp(new_s)
                for j in range(1):
                    if filled_new_s[j]['score'] < (filled_new_s[0]['score'] * (1 - 0.5)):
                        break
                    # candidate = filled_new_s[j]['sequence'].replace('<s> ', '').replace('<s>', '').replace('</s>', '')
                    if self.is_candidate_valid(s, candidate):
                        already_in_candidates = False
                        for c in bert_candidates:
                            if c[0] == candidate:
                                already_in_candidates = True
                                break
                        if not already_in_candidates:
                            candidate_score = pos_scores[i]
                            bert_candidates.append((candidate, candidate_score))

        return bert_candidates



    def get_valid_antonyms_candidates(self, s, s_arr, pos_scores):
        antonyms_candidates = []

        for i in range(len(s_arr)):
            if not self.to_modify():
                continue
            antonyms_time_start = time.time()
            syns = wordnet.synsets(s_arr[i].replace('.', ''))

            antonyms_time_end = time.time()
            self.times_dict["antonyms calc time"] += antonyms_time_end - antonyms_time_start
            self.times_dict["antonyms calc calls"] += 1

            antonyms = []
            for sy in syns:
                for l in sy.lemmas():
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())
            for j in range(len(antonyms)):
                copy = s_arr.copy()
                copy[i] = antonyms[j]
                candidate = ' '.join(copy)
                if self.is_candidate_valid(s, candidate):
                    candidate_score = pos_scores[i]
                    antonyms_candidates.append((candidate, candidate_score))

        return antonyms_candidates




    def select_neighbors(self, s):
        nlp = self.pipeline
        # sentence = pos_tokenizer(s)
        # s_arr = []
        # pos_scores = []
        # max_gpt_score = self.original_gpt_score * self.max_gpt_ratio
        # for i in range(len(sentence)):
        #     s_arr.append(sentence[i].text)
        #     pos = sentence[i].pos_
        #     if (pos in self.pos_scores_histogram.keys()) and (self.pos_scores_histogram[pos] > 0):
        #         pos_scores.append(self.pos_scores_histogram[pos])
        #     else:
        #         pos_scores.append(self.pos_scores_histogram["X"])
        candidates = []
        s_arr = s.split(" ")
        pos_scores = [0.5]*len(s_arr) #default
        if self.use_bert:
            candidates += self.get_valid_bert_candidates(s, s_arr, pos_scores)
        # 
        if self.use_antonyms:
            candidates += self.get_valid_antonyms_candidates(s, s_arr, pos_scores)

        if self.enable_word_deletion:
            candidates += self.get_valid_deleted_words_candidates(s, s_arr, pos_scores)

        #candidates.sort(key=lambda candidates:candidates[1])
        #candidates.reverse()
        sentences = [c[0] for c in candidates]
        scores = [c[1] for c in candidates]
        dist = [score/sum(scores) for score in scores]
        if self.use_pos_analysis:
            try:
                neighbors = list(np.random.choice(sentences, min(len(sentences),self.branching_factor), p=dist, replace=False))
            except: #sometimes sum(dist)!=1
                neighbors = [c[0] for c in candidates[:min(len(candidates), self.branching_factor)]]
        else:
            neighbors = list(np.random.choice(sentences, min(len(sentences), self.branching_factor), replace=False))

        return s_arr, neighbors




    def neighbors(self, node):
        """For a given node, returns (or yields) the list of its neighbors. this method must be implemented in a subclass"""

        neighbors_time_start = time.time()

        neighbors = []
        original = node.replace(' , ', ', ').replace(' .', '.').replace('<s> ', '').replace('<s>', '').replace('</s>', '')
        nlp = self.pipeline
        s, neighbors = self.select_neighbors(original)


        neighbors_time_end = time.time()
        self.times_dict["neighbors calc time"] += neighbors_time_end - neighbors_time_start
        self.times_dict["neighbors calc calls"] += 1
        return neighbors


    def is_goal_reached(self, current):
        """ returns true when we can consider that 'current' is the goal"""
        label, score = self.black_box.predict(current)
        return label != self.original_label and score >= self.confidence_threshold

    def reconstruct_path(self, last, reversePath=False):
        def _gen():
            current = last
            while current:
                yield current.data
                current = current.came_from
        if reversePath:
            return _gen()
        else:
            return reversed(list(_gen()))

    def choose_node_from_openset(self, openset):
        best = heappop(openset)
        return best
        # if not self.uniform_cost_epsilon:
        #     return best
        #
        # candidates = []
        # curr = best
        # while curr.gscore <= (best.gscore + self.epsilon):
        #     candidates.append(curr)
        #     if len(openset) > 0:
        #         curr = heappop(openset)
        #     else:
        #         break
        #
        # candidates = sorted(candidates, key=lambda c: self.heuristic_cost_estimate(c))
        #
        # for i in range(1, len(candidates)):
        #     heappush(openset, candidates[i])
        #
        # return candidates[0]


    def astar(self, start, reversePath=False):
        self.original_gpt_score = self.gpt2_score(start)
        self.max_distance = len(start.split(" "))
        if self.is_goal_reached(start):
            return [start]
        searchNodes = AStar.SearchNodeDict()
        startNode = searchNodes[start] = AStar.SearchNode(
            start, gscore=.0, fscore=0)
        openSet = []
        heappush(openSet, startNode)
        i = 0
        while openSet:
            start_pop_time = time.time()

            #i += 1
            #if i > self.max_open_set_pops:
            #    return None, i, self.times_dict
            current = self.choose_node_from_openset(openSet)
            if self.is_goal_reached(current.data):
                return self.reconstruct_path(current, reversePath), self.expensive_calls_counter, self.times_dict
            if self.expensive_calls_counter >= self.max_expensive_calls:
                return None, self.expensive_calls_counter, self.times_dict
            current.out_openset = True
            current.closed = True
            for neighbor in map(lambda n:searchNodes[n], self.neighbors(current.data)):
                if neighbor.closed:
                    continue
                #tentative_gscore = current.gscore + \
                #    self.distance_between(current.data, neighbor.data)
                tentative_gscore = current.gscore + \
                                   self.distance_between(start, neighbor.data) #we want to measure distance compared to the original sentence
                if tentative_gscore >= neighbor.gscore:
                    continue
                neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                neighbor.fscore = ((1-self.heuristic_weight)*tentative_gscore) + (self.heuristic_weight * self.heuristic_cost_estimate(neighbor))
                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)

            end_pop_time = time.time()
            self.times_dict["pop time"] += end_pop_time - start_pop_time
            self.times_dict["pops"] += 1

        return None, self.expensive_calls_counter, self.times_dict

