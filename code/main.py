# Header

import itertools
import json
import re

import evaluate
import numpy as np
import pandas as pd
import sklearn
import torch
import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


class Trainer(Trainer):
    loss_fn = torch.nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        loss = self.loss_fn(
            logits.view(-1, self.model.config.num_labels), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


class Dataset(torch.utils.data.Dataset):

    # The variable segmented_word will be a list of segmented words ex. un@bound@ed
    # This class creates a dataset where each item in self._items is one classification task (True or False). For each word decomposition there will be len(word)-1 classification tasks.
    # Therefor there are many more classifications tasks than there are words.
    def __init__(self, segmented_words):
        self._segmented_words = segmented_words
        self._items = {}

        # Loops over each segmented word.
        for w in segmented_words:
            # Splits the word by @ and returns a list of morphemes: "un@bound@ed" ---> ['un', 'bound', 'ed']
            segments = w.split("@")

            # Var word will represent the unsegmented word.
            word = "".join(segments)

            # in between characters, there are len-1 posibilities.
            # Creates a list of char lengs for each morpheme.
            segment_lens = [len(s) for s in segments]

            # This keeps track of where the gold segmentation indexes are in the string.
            # ['un', 'bound', 'ed'] ---> {2, 7}. A word bound could be inserted at either idx
            # 2 or 7 for a correct label.
            hyphen_pos = set(itertools.accumulate(segment_lens[:-1]))


            # For instance: word == "unbounded", range is 1, 9.
            for i in range(1, len(word)):
                text = f"{word} {word_separator_token} {word[:i]}{morph_boundary_token}{word[i:]}"

                # Checks if current value of i(1-len(word)) (representing current segmentation in text)
                # is a valid segmentation in comparison to the gold standard.
                # Returns 0 if False.
                # Returns 1 if True.
                label = int(i in hyphen_pos)
                
                value_dict = {"text": text, "label": label}
                value_dict.update(tokenizer(text))
                # Each item will be one word2example segmentation, label, attention mask 
                self._items[len(self._items)] = value_dict
        super()

    # Changed this function a bit since it threw an error otherwise. Changed ... classes=[0, 1] ----> classes=np.array([0, 1])
    # It was complaining that classes had to be an instance of an ndarray.
    # It returns a one-dimensional array with two weights. One weight per class (0, 1). The weights are calculated as such: n_samples / (n_classes * np.bincount(y))
    # There are later used to update the loss function torch.nn.CrossEntropyLoss which takes the argument weights
    # These is useful to do when there is an unbalanced dataset which this will be
    # This is because there are many more False labels than there are True labels.
    def get_class_weights(self):
        weights = sklearn.utils.class_weight.compute_class_weight(
            "balanced", classes=np.array([0, 1]), y=[i["label"] for i in self._items.values()]
        )
        return weights

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]
    

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return clf_metrics.compute(predictions=predictions, references=labels)


def eval_word(ground, predicted):
    tp = fp = fn = 0
    gi = pi = 0
    while gi < len(ground) and pi < len(predicted):
        g = ground[gi]
        p = predicted[pi]
        if g == p:
            if g == "@":
                tp += 1
            gi += 1
            pi += 1
        elif g == "@":
            fn += 1
            gi += 1
        elif p == "@":
            fp += 1
            pi += 1
        else:
            assert False, (ground, predicted)
    assert gi == len(ground) and pi == len(predicted)
    return tp, fp, fn


def _esc_spec(c):
    if c == " ":
        return "_"
    return c


def test_model(model, test_data):
    TP = FP = FN = 0
    ACC = 0
    predictions = []

    model.eval()
    for ground in tqdm.tqdm(test_data):
        word = "".join(ground.split("@"))
        ww = []
        wit = iter(word)
        candidates = [
            f"{word} {word_separator_token} {word[:i]}{morph_boundary_token}{word[i:]}"
            for i in range(1, len(word))
        ]
        tokenized = tokenizer(candidates, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            res = model(**tokenized).logits.argmax(1).cpu().tolist()
        for i in res:
            c = _esc_spec(next(wit))
            ww.append(c)
            if i == 1:
                ww.append("@")
        c = _esc_spec(next(wit))
        ww.append(c)
        predicted = "".join(ww)
        predictions.append(predicted)
        if ground == predicted:
            ACC += 1
        tp, fp, fn = eval_word(ground, predicted)
        TP += tp
        FP += fp
        FN += fn
    return ACC, TP, FP, FN, predictions

# Name the data as: 
# train.iu.csv
# dev.iu.csv
# test.iu.csv

# These should have been turned into csv-files using the format_data.py file.
def load_lang_data(path2data_dir, lang):
    data_train = pd.read_csv(path2data_dir + f"train.{lang}.csv", sep="\t", header=None)
    data_dev = pd.read_csv(path2data_dir + f"dev.{lang}.csv", sep="\t", header=None)
    data_test = pd.read_csv(path2data_dir + f"test.{lang}.csv", sep="\t", header=None)
    return data_train, data_dev, data_test


if __name__=="__main__":

    # Vars
    path2data_dir = "/home/mathias/Desktop/HI/hpc/inuktitut/llm_segm/reimplementation/"
    path2out_dir = "/home/mathias/Desktop/HI/hpc/inuktitut/llm_segm/reimplementation/out/"
    lang = "iu"
    model_name = "cis-lmu/glot500-base"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    word_separator_token = "<__word-separator>"
    morph_boundary_token = "<__morph-boundary>"
    assert tokenizer.add_tokens([morph_boundary_token, word_separator_token])

    # Model instantiation, embedding resizing and weight resetting.
    model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            )
    embs = model.resize_token_embeddings(len(tokenizer))
    embs.weight.data[-1] = 0  # embs.weight[238].detach()
    embs.weight.data[-2] = 0  # embs.weight[2203].detach()

    # Loads train, dev and test data as pandas dataframes from csv. unbounded,un@bound@ed
    data_train, data_dev, data_test = load_lang_data(path2data_dir, lang)

    # Instantiates dataset from the train, dev dataframes.
    # For example; data_train[1] will correspond to the column with the separated words. [0] the full word.
    d_train = Dataset(data_train[1].tolist())
    d_val = Dataset(data_dev[1].tolist())

    # Loading the evaluation metrics
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    # Instantiating the collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Default training arguments based on the code/paper
    training_args = TrainingArguments(
            output_dir=path2out_dir + "glot500-iu-morph",
            learning_rate=2e-5,
            per_device_train_batch_size=256,
            per_device_eval_batch_size=256,
            num_train_epochs=30,
            logging_steps=250,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            metric_for_best_model="f1",
            greater_is_better=True,
            load_best_model_at_end=True,
            warmup_steps=20,
            )
    
    # Push the model to GPU, 
    model.cuda()
    model.train()

    # Instantiating the custom Trainer class.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=d_train,
        eval_dataset=d_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Fetching class weights and updates loss function?
    class_weights = d_train.get_class_weights()
    trainer.loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights.astype("float32"))
    ).to(model.device)

    # Initiates training
    trainer.train()

################----- EVALUATION STARTS HERE -----##############
################----- MOVE TO SEPARATE FILE ------##############
    
    # Initiates evaluation mode for model.

    model.eval()

    ACC, TP, FP, FN, predictions = test_model(model, data_test[1].tolist())

    predictions = [re.sub("@", " ", p) for p in predictions]
    df = pd.DataFrame({"word": data_test[0].tolist(), "predictions": predictions})
    df.to_csv(f"{path2out_dir}preds/{lang}.pred", header=None, index=False, sep="\t")

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = 2 * P * R / (P + R)


    with open(f"{path2out_dir}results/results.{lang}.json", "w") as results_f:
            results_map = {
                "ACC": np.round(100 * ACC / len(data_test), 2),
                "Prec": np.round(100 * P, 2),
                "Rcl": np.round(100 * R, 2),
                "F1": np.round(100 * F1, 2),
                "train-len": len(data_train),
                "dev-len": len(data_dev),
                "test-len": len(data_test),
                "LANG": lang,
            }
            json.dump(results_map, results_f, indent=4)
