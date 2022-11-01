from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from transformers import TrainingArguments
import sys
import os
from datasets import Dataset, DatasetDict
from transformers import Trainer
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import csv
import time
import argparse
from transformers import EarlyStoppingCallback


sys.path.append(os.getcwd())  # Should add main repo dir to paths
from src.utils.load_data import Data_Manager


def sliding_window_ds_approach(df, window_size=50):
    """Reformat dataset entry into the sliding window approach.
    Concanate explanation with question.
    Multiple entries are grouped with a question index

    Args:
        df (pandas dataframe): dataset
        window_size (int, optional): number of characters to add from the question. Defaults to 50.

    Returns:
        pandas dataframe: resulting dataset
    """
    dp = (
        []
    )  # table_of_content=["question", "answer", "label", "analysis", "complete analysis", "explanation", "idx"]
    for _, row in df.iterrows():
        (
            question,
            answer,
            label,
            analysis,
            completeanalysis,
            explanation,
            idx,
            idx_complete,
        ) = row
        cache = explanation + " | " + question
        cache = cache.split()  # Split on whitespace
        if len(cache) <= window_size:
            dp.append(
                (
                    explanation + " | " + question,
                    answer,
                    label,
                    analysis,
                    idx,
                    idx_complete,
                )
            )
        else:
            while len(cache) > window_size:
                sentence1 = " ".join(cache[: 150 + window_size])
                dp.append((sentence1, answer, label, analysis, idx, idx_complete))
                cache = cache[150:]

    return pd.DataFrame(
        dp, columns=["question", "answer", "label", "analysis", "idx", "idx_complete"]
    )


def sliding_window_ds_approach_keep_question(df, window_size=50):
    """Reformat dataset entry into the sliding window approach.
    For each sample use the question and pad the rest with explanation.
    Multiple entries are grouped with a question index

    Args:
        df (pandas dataframe): dataset
        window_size (int, optional): number of characters to add from the question. Defaults to 50.

    Returns:
        pandas dataframe: resulting dataset
    """

    dp = (
        []
    )  # table_of_content=["question", "answer", "label", "analysis", "complete analysis", "explanation", "idx"]
    for index, row in df.iterrows():
        (
            question,
            answer,
            label,
            analysis,
            completeanalysis,
            explanation,
            idx,
            idx_complete,
        ) = row
        question = question + " | "
        question_len = len(question.split())
        cache = explanation
        cache = cache.split()  # Split on whitespace
        if len(cache) <= window_size:
            dp.append(
                (question + explanation, answer, label, analysis, idx, idx_complete)
            )
        else:
            while len(cache) > window_size:
                append_len = 400 - question_len
                append_len = append_len if append_len > 0 else 0
                if append_len <= 0:  # Debugging
                    print(
                        "ERROR: Append len is 0. Question: %s len question %s"
                        % (question, question_len)
                    )
                    exit()
                sentence1 = question + " ".join(cache[: append_len + window_size])
                dp.append((sentence1, answer, label, analysis, idx, idx_complete))
                cache = cache[append_len:]

    return pd.DataFrame(
        dp, columns=["question", "answer", "label", "analysis", "idx", "idx_complete"]
    )


def create_dataset(sliding_window=""):
    """transform pandas dataset into transformer datasets.dataset

    Args:
        sliding_window (str, optional): Token limit avoiding approaches. Defaults to "".

    Returns:
        datasets.dataset: dataset
    """
    dm = Data_Manager()
    train_data, dev_data, test_data = dm.get_dataset_as_df("tdt_rational")

    train_data["idx"] = [i for i in range(train_data["idx"].size)]
    dev_data["idx"] = [i for i in range(dev_data["idx"].size)]
    test_data["idx"] = [i for i in range(test_data["idx"].size)]

    # Create second idx column for each dataset to be used for evaluation later
    train_data["idx_complete"] = [i for i in range(train_data["idx"].size)]
    dev_data["idx_complete"] = [i for i in range(dev_data["idx"].size)]
    test_data["idx_complete"] = [i for i in range(test_data["idx"].size)]

    train_data["label"] = [int(i) for i in train_data["label"]]
    dev_data["label"] = [int(i) for i in dev_data["label"]]
    test_data["label"] = [int(i) for i in test_data["label"]]

    if sliding_window == "simple":
        train_data = sliding_window_ds_approach(train_data)
        dev_data = sliding_window_ds_approach(dev_data)
        test_data = sliding_window_ds_approach(test_data)

        train_data["idx"] = [i for i in range(train_data["idx"].size)]  # Reset index
        dev_data["idx"] = [i for i in range(dev_data["idx"].size)]
        test_data["idx"] = [i for i in range(test_data["idx"].size)]

    elif sliding_window == "keep_question":
        train_data = sliding_window_ds_approach_keep_question(train_data)
        dev_data = sliding_window_ds_approach_keep_question(dev_data)
        test_data = sliding_window_ds_approach_keep_question(test_data)

        train_data["idx"] = [i for i in range(train_data["idx"].size)]  # Reset index
        dev_data["idx"] = [i for i in range(dev_data["idx"].size)]
        test_data["idx"] = [i for i in range(test_data["idx"].size)]

    dataset_train = Dataset.from_pandas(train_data)
    dataset_dev = Dataset.from_pandas(dev_data)
    dataset_test = Dataset.from_pandas(test_data)

    complete_ds = DatasetDict(
        {"train": dataset_train, "dev": dataset_dev, "test": dataset_test}
    )  
    return complete_ds


def tokenize_function(example):
    return tokenizer(
        example["question"], example["answer"], truncation=True, max_length=512
    )  

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    print([(l, p) for l, p in list(zip(labels, preds)) if l == 1][:10])
    # calculate accuracy using sklearn's function
    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    print("dev set len:", len(tokenized_datasets["dev"]), "pred len:", len(preds))
    print("F1:", f1, "Acc:", acc)
    return {
        "f1_score": f1,
    }


def combine_splitted_samples(dataset):
    """split the dataset in many datasets where all samples are combined with the same idx_complete

    Args:
        dataset (huggingface datasets): dataset to be splitted

    Returns:
        list: list of hugginface datasets
    """
    ds_list = []
    temp_dict = {}
    for sample in dataset:
        if sample["idx_complete"] not in temp_dict:
            temp_dict[sample["idx_complete"]] = dict(
                map(lambda x: (x, []), sample.keys())
            )
        for key in temp_dict[sample["idx_complete"]].keys():
            temp_dict[sample["idx_complete"]][key].append(sample[key])

    for sample in temp_dict.values():
        ds_list.append(Dataset.from_dict(sample))

    return ds_list


def predict_on_dataset(list_dataset):
    """Predict all samples in a list of datasets and take the mean of the predictions as the final prediction

    Args:
        list_dataset (list): list of small datasets containing all sub samples of a complete sample

    Returns:
        tuple: prediction and correct label as two lists
    """
    preds = []
    gt = []
    for sub_dataset in list_dataset:
        sub_predicition = trainer.predict(sub_dataset)
        # calc the average of the predictions
        sub_predicition = sub_predicition.predictions.mean(axis=0)
        preds.append(sub_predicition.argmax())
        gt.append(sub_dataset[0]["label"])
    return preds, gt


if __name__ == "__main__":
    model_base_path = os.path.join(os.getcwd(), "data", "done", "model_output")

    parser = argparse.ArgumentParser(description="Evaluate LegalBert")
    parser.add_argument(
        "--dataset_type",
        type=str,
        help='Dataset type to use. choose from: "", "simple", or "keep_question"',
        default="",
    )
    parser.add_argument(
        "--model",
        type=str,
        help='BERT based model to use',
        default="nlpaueb/legal-bert-base-uncase",
    )
    args = parser.parse_args()

    legal_ds = create_dataset(args.dataset_type)  # "", "simple" or "keep_question"

    checkpoint = args.model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    tokenized_datasets = legal_ds.map(tokenize_function, batched=True)
    samples = tokenized_datasets["train"]["input_ids"][:8]

    N = 3
    results = []
    for index in range(N):
        training_args = TrainingArguments(
            output_dir=os.path.join(model_base_path, checkpoint + "_finetune"),
            group_by_length=True,
            per_device_train_batch_size=4,  
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=100,
            fp16=True,
            learning_rate=3e-5,
            warmup_steps=10,
            gradient_accumulation_steps=2,
            logging_strategy="epoch",
            seed=index,
            load_best_model_at_end=True,
            greater_is_better=True,
            metric_for_best_model="f1_score",
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=2
        )

        trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["dev"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=-0.05)]

        )

        trainer.train()

        # Eval
        # Combine all samples with the same idx_complete into a single batch for testing
        dev_dataset_list = combine_splitted_samples(tokenized_datasets["dev"])
        test_dataset_list = combine_splitted_samples(tokenized_datasets["test"])
        # predict over dev dataset list and combine for each list entry all predictions to a final prediction
        preds, gt = predict_on_dataset(dev_dataset_list)

        # Score dev set
        print("F1 Score (Macro)", f1_score(preds, gt, average="macro"))
        print("F1 Score (binary)", f1_score(preds, gt, average="micro"))
        print("Accuracy", accuracy_score(preds, gt))
        results.append(
            (
                f1_score(preds, gt, average="macro"),
                f1_score(preds, gt, average="micro"),
                accuracy_score(preds, gt),
            )
        )

        # Score test set
        results2 = []
        predictions = trainer.predict(tokenized_datasets["test"])
        preds = np.argmax(predictions.predictions, axis=-1)
        print(
            "F1 Score (Macro)",
            f1_score(preds, tokenized_datasets["test"]["label"], average="macro"),
        )
        print(
            "F1 Score (binary)",
            f1_score(preds, tokenized_datasets["test"]["label"], average="micro"),
        )
        print("Accuracy", accuracy_score(preds, tokenized_datasets["test"]["label"]))
        results2.append(
            (
                f1_score(preds, tokenized_datasets["test"]["label"], average="macro"),
                f1_score(preds, tokenized_datasets["test"]["label"], average="micro"),
                accuracy_score(preds, tokenized_datasets["test"]["label"]),
            )
        )

    path = os.path.join(
        os.path.join(os.getcwd(), "data", "done"), checkpoint + "_finetune"
    )

    if args.dataset_type == "":
        run_name = "FT_QEA_DEV_TESTCONFS_" + str(int(time.time()))
    elif args.dataset_type == "simple":
        run_name = "FT_QEA_DEV_SWS_TESTCONFS_" + str(int(time.time()))
    elif args.dataset_type == "keep_question":
        run_name = "FT_QEA_DEV_SWA_TESTCONFS_" + str(int(time.time()))

    # Calc mean F1 macro and binary score
    f1_macro = np.mean([i[0] for i in results])
    f1_binary = np.mean([i[1] for i in results])
    accuracy = np.mean([i[2] for i in results])

    print(
        "Mean N=%s Score: F1 Score (Macro)" % N,
        f1_macro,
        "F1 Score (binary)",
        f1_binary,
        "Accuracy",
        accuracy,
    )
    f1_macro = np.mean([i[0] for i in results2])
    f1_binary = np.mean([i[1] for i in results2])
    accuracy = np.mean([i[2] for i in results2])
    print(
        "2 Approach: Mean N=%s Score: F1 Score (Macro)" % N,
        f1_macro,
        "F1 Score (binary)",
        f1_binary,
        "Accuracy",
        accuracy,
    )
    results.append((f1_macro, f1_binary, accuracy))

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, run_name), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["f1_macro", "f1_binary", "accuracy"])
        writer.writerows(results)
