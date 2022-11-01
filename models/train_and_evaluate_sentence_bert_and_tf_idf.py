from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

sys.path.append(os.getcwd())  # Should add main repo dir to paths
from models import train_and_evaluate_transformer_classifier as train_and_evaluate
from nltk.tokenize import sent_tokenize


def transform_data(df):
    """Parse the dataframe to find all answers for each question for tf-idf similarity calculation.

    Args:
        df (pandas.df): dataset as pandas df

    Returns:
        dict: dict where the question is the key and all answers are in a list as the value
    """
    transform_dict = {}
    for question in df["question"].unique():
        sub_df = df[["answer", "label"]][df["question"] == question]
        transform_dict[question] = sub_df.values.tolist()
    return transform_dict


def calc_tf_idf(question, answer_list, vectorizer):
    """Fit the TfIDF Vectorizer by sklearn on the question.
    Calculate the sim of all answers to the question vector.


    Args:
        question (string): dictionary key
        answer_list (list): list of lists with [answer, label]
        vectorizer (TfidfVectorizer): vectorizer
    """

    score_list = []
    for answer in answer_list:
        label = answer[1]
        answer_text = answer[0]
        score_list.append(
            (
                vectorizer.transform([question,]).dot(
                    vectorizer.transform(
                        [
                            answer_text,
                        ]
                    )
                ),
                label,
            )
        )
    print(score_list)


def tf_idf_similarity():
    """
    Calculate tf-idf similarity for all questions in a dataframe.
    """
    train_df, dev_df, test_df = train_and_evaluate.create_dataset("simple", as_df=True)

    train_dict = transform_data(train_df)
    tfidf_vectorizer = TfidfVectorizer(input="content", stop_words="english")
    tfidf_vectorizer = tfidf_vectorizer.fit(train_dict.keys())
    for question in train_dict.keys():
        calc_tf_idf(question, train_dict[question], tfidf_vectorizer)


def sentence_bert_similarity(train_df, type="minimal"):
    """Separate each entry into smaller chunks
    Calculate best answer based on sentence bert cosine distance.
    Nearest answer chunk is considered as correct

    Args:
        train_df (pandas.df): dataset to evaluate
        type (str, optional): Sliding window approach. Defaults to "minimal".
    """

    checkpoint = "sentence-transformers/all-mpnet-base-v2"
    model = SentenceTransformer(checkpoint)

    train_dict = transform_data(train_df)
    min_prediction = []
    buffer = []
    for question in train_dict.keys():
        question_sentences = sent_tokenize(question)
        for answer in train_dict[question]:
            answer_sentences = sent_tokenize(answer[0])
            label = answer[1]

            for question_sentence in question_sentences:
                embeddings = model.encode([question_sentence] + answer_sentences)
                cos_sim = cosine_similarity(
                    [embeddings[0]],
                    embeddings[1:],
                )[0]

                if type == "minimal":
                    buffer.append((min(cos_sim), label))
                elif type == "average":
                    buffer.append((sum(cos_sim) / len(cos_sim), label))
                else:
                    print(
                        "False Sliding Window Stragy. Choose between 'minimal' and 'average'"
                    )

        min_prediction.append(sorted(buffer, key=lambda x: x[0])[0][1])
        buffer = []

    print("Sentence Bert Similarity Minimal")
    correct_min_predictions = sum([p for p in min_prediction if p == 1])
    print(
        f"Correct ({type} cos_sim) Prediction: {correct_min_predictions}/{len(min_prediction)} = {correct_min_predictions/len(min_prediction)}"
    )


def check_for_cues(df):
    """Try to find Unigrams/Bigrams which have a connection to correct or incorrect answers. Use TF-IDF with 2 docs (correct, incorrect)

    Args:
        df (pandas.dataframe): dataset
    """

    # Get the correct and incorrect answers
    correct_answers = df[df["label"] == 1]["answer"].tolist()
    incorrect_answers = df[df["label"] == 0]["answer"].tolist()
    print(len(correct_answers), len(incorrect_answers))
    print(correct_answers[:5])
    correct_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), stop_words="english", max_features=10
    )
    incorrect_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), stop_words="english", max_features=10
    )

    print("Correct Answers cue candidates:")
    X_c = correct_vectorizer.fit_transform(correct_answers).todense()
    df_c = pd.DataFrame(X_c, columns=correct_vectorizer.get_feature_names())
    df_c = df_c.mean(axis=0).sort_values(ascending=False)
    print(df_c)

    print("Incorrect Answers cue candidates:")
    X_i = incorrect_vectorizer.fit_transform(incorrect_answers).todense()
    df_i = pd.DataFrame(X_i, columns=incorrect_vectorizer.get_feature_names())
    df_i = df_i.mean(axis=0).sort_values(ascending=False)
    print(df_i)

    # Compare df_c and df_i for common words
    common_words = set(df_c.index) & set(df_i.index)
    print("Most Important Uni-, or Bigrams that are in both sets", common_words)
    # Compare df_c and df_i for uncommon words
    uncommon_words = set(df_c.index) - set(df_i.index)

    print(
        "Most Important Uni-, or Bigrams that are in only one of the two sets",
        uncommon_words,
        set(df_i.index) - set(df_c.index),
    )


if __name__ == "__main__":
    model_base_path = os.path.join(os.getcwd(), "data", "done", "model_output")

    train_df, dev_df, test_df = train_and_evaluate.create_dataset("", as_df=True)

    print("Check for cues ")
    check_for_cues(pd.concat([train_df, dev_df, test_df]))

    print("Train set results ####################")
    sentence_bert_similarity(train_df, "minimal")
    sentence_bert_similarity(train_df, "average")

    print("Dev set results ####################")
    sentence_bert_similarity(dev_df, "minimal")
    sentence_bert_similarity(dev_df, "average")

    print("Test set results ####################")
    sentence_bert_similarity(test_df, "minimal")
    sentence_bert_similarity(test_df, "average")
