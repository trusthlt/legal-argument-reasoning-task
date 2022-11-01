import os
import csv
import random
import pandas as pd


def read_file(path):
    """Read the csv file containing the data entries

    Args:
        path (string): path to file

    Returns:
        list: dataset entries
    """
    with open(path, "r") as f:
        csvreader = csv.reader(f, delimiter=",", quotechar='"')
        next(
            csvreader
        )  # header: question,answer,label,analysis,complete analysis,explanation,idx
        entries = list(csvreader)
    return entries


class Data_Manager:
    """Class to read the data and create the datasplits used fore evaluation"""

    def __init__(
        self, data_path=os.path.join(os.getcwd(), "data", "processed", "done")
    ) -> None:
        self.data_path = data_path
        print(self.data_path)

    def _load_files(self, delimiter="\t", files_only=False):
        """Load the entries from each chapter file into one dictionary

        Args:
            delimiter (string, optional): Separation Character of csv file. Defaults to "\t".
            files_only (bool, optional): return file names if true. Defaults to False.

        Returns:
            dict: dataset
        """
        files = [
            (f, os.path.join(os.getcwd(), self.data_path, f))
            for f in os.listdir(self.data_path)
        ]
        if files_only:
            return files

        data = {}
        for file_name, file_path in files:
            with open(file_path, "r") as f:
                reader = csv.reader(f, delimiter=delimiter)
                data[file_name] = list(reader)[1:]

        return data

    def get_split_dataset(self, split_type):
        """
        Split the dataset accordingly to the split_type. Return three lists of entries.
        params:
            split_type: string, either "tdt_random", "tdt_rational" or "cross_validation"
        """
        print("splittype", split_type)
        if split_type == "tdt_rational":
            if os.path.exists(
                os.path.join("data", "done", "dataset", "train.csv")
            ):  # Shortcut for exisiting datasplit files
                train_data = read_file(
                    os.path.join("data", "done", "dataset", "train.csv")
                )
                dev_data = read_file(os.path.join("data", "done", "dataset", "dev.csv"))
                test_data = read_file(
                    os.path.join("data", "done", "dataset", "test.csv")
                )
            else:
                train_data, dev_data, test_data = [], [], []
                for key in sorted(list(data_dict.keys())):
                    ch_len = len(data_dict[key])
                    for index, dp in enumerate(data_dict[key]):
                        if index < int(0.8 * ch_len):
                            train_data.append(dp)
                        elif index < int(0.9 * ch_len):
                            dev_data.append(dp)
                        else:
                            test_data.append(dp)
        elif split_type == "tdt_random":
            data_dict = self._load_files()
            dp_list = []
            for key in data_dict.keys():
                dp_list += data_dict[key]

            ds_len = len(dp_list)
            random.shuffle(dp_list)
            train_len = int(ds_len * 0.8)
            dev_test_len = int(ds_len * 0.1)
            train_data = dp_list[:train_len]
            dev_data = dp_list[train_len : train_len + dev_test_len]
            test_data = dp_list[train_len + dev_test_len :]

        elif split_type == "cross_validation":
            dp_list = []
            for key in data_dict.keys():
                dp_list += data_dict[key]
            ds_len = len(dp_list)

            train_len = int(ds_len * 0.9)

            train_data = dp_list[:train_len]
            test_data = dp_list[train_len:]
            dev_data = []
        else:
            raise ValueError("Invalid dataset split type")

        return train_data, dev_data, test_data

    def get_dataset_as_df(
        self,
        split_type,
        table_of_content=[
            "question",
            "answer",
            "label",
            "analysis",
            "complete analysis",
            "explanation",
            "idx",
        ],
    ):
        """Transform dataset into a pandas dataframe

        Args:
            split_type (strign): type of dataset split (tdt_rational/tdt_random/cross_validation)
            table_of_content (list, optional):header. Defaults to ["question", "answer", "label", "analysis", "complete analysis", "explanation", "idx"].
        """

        def add_idx_col(data):
            for index, dp in enumerate(data):
                dp.append(str(index))
            return data

        train, dev, test = self.get_split_dataset(split_type)
        train = add_idx_col(train)
        dev = add_idx_col(dev)
        test = add_idx_col(test)

        train_df = pd.DataFrame(train)
        train_df.columns = table_of_content

        dev_df = pd.DataFrame(dev)
        dev_df.columns = table_of_content

        test_df = pd.DataFrame(test)
        test_df.columns = table_of_content

        return train_df, dev_df, test_df

    def get_file_names(self):
        return self._load_files(files_only=True)


if __name__ == "__main__":
    dm = Data_Manager()
    train, dev, test = dm.get_dataset_as_df("tdt_rational")

    train.to_csv("data/processed/train.csv", index=False)
    dev.to_csv("data/processed/dev.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
