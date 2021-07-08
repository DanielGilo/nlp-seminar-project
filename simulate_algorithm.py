import pickle
import numpy as np
import editdistance as ed
import pandas as pd
import classifiers


def get_baseline_cf(predicted_x, sentence_label, original_sentence):
    min_dist = np.inf
    conf_level = 0
    cf = None
    for i in range(len(predicted_x)):
        new_sentence = predicted_x["text"].loc[i].replace(' , ', ', ').replace(' .', '.').replace('.','').lower()
        predicted_y, prob = predicted_x["label"].loc[i], predicted_x["confidence"].loc[i]
        d = ed.eval(original_sentence.split(" "), new_sentence.split(" "))
        if (0 < d < min_dist) and (predicted_y != sentence_label):
            cf = new_sentence
            min_dist = d
            conf_level = prob
    return cf, min_dist, conf_level


if __name__ == '__main__':
    ranges = list((i, i+49) for i in range(0, 151, 50))
    #ranges = [(0, 49)]
    budgets = list(range(50, 2001, 50))
    hw_diff = 0.25
    ds_name = "SST"
    classifier = classifiers.CustomClassifier("models/{}/".format(ds_name))
    ds = pd.read_csv("datasets/{}/predicted_train.csv".format(ds_name), header=0)
    train_x = ds["text"]
    min_hw = 0.0
    dir = "results/Paper/{}/specific+agnostic/".format(ds_name)
    for budget in budgets:
        results_list = []
        for _range in ranges:
            # if (ds_name == "Amazon") and (_range[0] == 80):
            #     continue
            found_non_baseline_counter = 0
            dist_sum = 0
            hw = 1.0
            with open("{}hw={}_start_{}_end_{}_results.pkl".format(dir, hw, _range[0],
                                                                                                  _range[1]), 'rb') as f:
                arr = pickle.load(f)
            for i in range(_range[1] - _range[0] + 1):
                hw = 1.0
                found_non_baseline = False
                spent = 0
                original_sentence = arr[i]["original_sentence"].replace(' , ', ', ').replace(' .', '.').replace('.','').lower()
                original_predicted_label = arr[i]["original_predicted_label"]
                cf, dist, conf = get_baseline_cf(ds, original_predicted_label, original_sentence)
                results = {"sentence_index": _range[0] + i, "confidence_level": conf, "distance": dist,
                    "language_model_ratio": 1, "original_predicted_label": original_predicted_label,
                           "original_sentence": original_sentence,
                       "counterfactual": cf}
                while (budget - spent) > 0:
                    with open("{}hw={}_start_{}_end_{}_results.pkl".format\
                                  (dir, hw, _range[0], _range[1]), 'rb') as f:
                        arr = pickle.load(f)
                    if (arr[i]["success"]) and (arr[i]["expensive_calls"] < (budget - spent)) and (arr[i]["distance"] <= dist):
                        dist = arr[i]["distance"]
                        found_non_baseline = True
                        results["confidence_level"] = arr[i]["confidence_level"]
                        results["distance"] = dist
                        results["language_model_ratio"] = arr[i]["language_model_ratio"]
                        results["counterfactual"] = arr[i]["counterfactual"]
                    spent += arr[i]["expensive_calls"]
                    hw -= hw_diff
                    if hw < min_hw:
                        break
                results_list.append(results)
        with open("{}budget={}_results.pkl".format(dir, budget), 'wb') as f1:
            pickle.dump(results_list, f1)
