import argparse
import glob
import os
import re
import time
import warnings
import random
from typing import List

import pandas as pd
from loguru import logger
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)

"""
Script to prepare notes data for LM. This script will load in the raw data file,
do some cleaning and write to a text file iteratively.

Example usage from cmd line for sampling:

```
python prepare_notes_for_lm.py --sample --train_sample_size 1000
 --test_sample_size 100
```
"""



class LMTextData:
    def __init__(
            self,
            train_test_notes_path=None,
            num_files_for_training=None,
            save_path=None,
            admin_language=None,
            replacement_map=None,
            remove_punctuation=True,
            sample=True,
            train_sample_size=500,
            test_sample_size=100,
            chunk_size=128,
            seed=41,
            text_col=None,
    ):
        self.admin_language = admin_language
        self.sample = sample
        self.train_sample_size = train_sample_size
        self.test_sample_size = test_sample_size
        self.train_test_notes_path = train_test_notes_path
        self.num_files_for_training = num_files_for_training
        self.save_path = save_path
        self.admin_language = admin_language
        self.chunk_size = chunk_size
        self.seed = seed
        self.text_col = text_col
        self.remove_punctuation = remove_punctuation
        self.replacement_map = replacement_map

        # make save_dirs if not exist
        for dir in [self.save_path]:
            if not os.path.exists(f"{dir}"):
                os.makedirs(f"{dir}")

    # - THIS IS NOW HANDELED BY THE LM TRAINING SCRIPT AND WILL NOT BE IMPLEMENTED BY
    # THE SCRIPT ANYMORE -
    # define a chunk function for each long text

    def chunks(self, sentence, chunk):
        return [sentence[i: i + chunk] for i in range(0, len(sentence), chunk)]

    # --------------------------------------------------------------------------------

    def clean_data(
            self,
            admin_language,
            notes_df: pd.DataFrame,
            min_tokens=5,
            replacement_map=None,
            remove_punctuation=False,
    ) -> pd.DataFrame:
        """
         TODO - add function to allow mapping of known acronyms to full words

         Basic cleaning of the text with some standardisation of punctuation and
         removal of certain characters.

         Args:
             text_col (string): name of column with text in
             notes_df (pd.DataFrame): CRIS produced clinical notes with following
             possible cols:


        Returns: pd.DataFrame: notes_df, filtered of redundant text
        """

        logger.info("Cleaning data!")

        text_col = self.text_col
        filtered_df = notes_df.copy()

        # remove rows with no data/clinical date data

        # remove some punctuation
        if remove_punctuation:
            filtered_df[text_col] = filtered_df[text_col].replace(
                r"\[.*?\]", "", regex=True
            )

        # print("Cleaning and applying some regex rules!")
        # remove refundnant new lines etc
        # nan_value = float("NaN")

        # heuristic rules
        for original, replacement in tqdm(self.replacement_map):
            # drop if text is none
            filtered_df = filtered_df[filtered_df[text_col].notna()]

            # in case dataframe is empty after drop
            if filtered_df.empty:
                return filtered_df

            filtered_df[text_col] = filtered_df[text_col].str.replace(
                original, replacement
            )

        # if admin language is not none, try remove?
        if admin_language is not None:
            for admin_token in admin_language:
                filtered_df[text_col] = filtered_df[text_col].str.replace(
                    admin_token, " "
                )
        # strip white space and lower case
        filtered_df[text_col] = filtered_df[text_col].str.strip()
        filtered_df[text_col] = filtered_df[text_col].str.lower()

        # found white spaces still persist - so now double remove them
        filtered_df[text_col] = [re.sub(r"\\s+", " ", x) for x in filtered_df[text_col]]

        # drop na

        return filtered_df.dropna()

    def combine_input_files(self, filenames):
        dfs_list = []
        for file in filenames:
            df = pd.read_csv(
                file, nrows=self.train_sample_size
            )
            print(len(df.index))
            dfs_list.append(df)
        combined_dfs = pd.concat(dfs_list, axis=0, ignore_index=True)
        print(combined_dfs.shape)

        return combined_dfs

    def read_write_all_text(self):
        """
        Function to read in and clean text files in preparation for transformer based
        language modelling

        """

        start = time.time()

        text_col = self.text_col
        # set up parameters based on whether its training data or test data
        # TODO - refactor to just do both train and test in one run without these
        # extra arguments for whether or not it is training or test data

        logger.info(f"working on training data: {self.train_test_notes_path}")
        if self.sample:
            logger.info("Will be sampling the datasets")

            train_save_path = (
                f"{self.save_path}/training_all_text_{self.train_sample_size}.txt"
            )

            test_save_path = (
                f"{self.save_path}/test_all_text_{self.test_sample_size}.txt"
            )
            # now load in the dataframe
            filenames = glob.glob(self.train_test_notes_path + "/*.csv")
            #train_filenames = random.sample(filenames, k=self.num_files_for_training, )
            if self.num_files_for_training < len(filenames):
                train_filenames = filenames[:self.num_files_for_training]
            else:
                raise Exception("Num_files_for_training includes all files, no data for Test")
            train_notes_temp = self.combine_input_files(filenames=train_filenames)
            test_filenames = [x for x in filenames if x not in train_filenames]
            test_notes_temp = self.combine_input_files(filenames=test_filenames)
        else:
            train_save_path = f"{self.save_path}/training_all_text.txt"
            test_save_path = f"{self.save_path}/test_all_text.txt"

            filenames = glob.glob(self.train_test_notes_path + "/*.csv")
            #train_filenames = random.sample(filenames, k=self.num_files_for_training)
            if self.num_files_for_training < len(filenames):
                train_filenames = filenames[:self.num_files_for_training]
            else:
                raise Exception("Num_files_for_training includes all files, no data for Test")
            train_notes_temp = self.combine_input_files(filenames=train_filenames)
            test_filenames = [x for x in filenames if x not in train_filenames]
            test_notes_temp = self.combine_input_files(filenames=test_filenames)


        train_filename_roots = [os.path.basename(x) for x in train_filenames]
        test_filename_roots = [os.path.basename(x) for x in test_filenames]
        print(f"For training using {self.num_files_for_training} files: {train_filename_roots} \n For test using files: {test_filename_roots}")
        logger.info(
            (
                f"Size of training notes data is: {train_notes_temp.shape} and test "
                f"data: {test_notes_temp.shape}"
            )
        )

        # TRAINING DATA #
        logger.info("Cleaning and writing training data to file!")
        # open training file to write to
        open_file = open(f"{train_save_path}", "w", encoding="utf8")

        # clean notes
        cleaned_training_notes = self.clean_data(
            admin_language=self.admin_language,
            remove_punctuation=self.remove_punctuation,
            replacement_map=self.replacement_map,
            notes_df=train_notes_temp,
        )

        # write text data alone for LM modelling etc
        cleaned_training_text = cleaned_training_notes[text_col].to_list()
        # write to file
        open_file.write("\n".join(cleaned_training_text))

        # TEST DATA #
        logger.info("Cleaning and writing test data to file!")
        # open test file to write to
        open_file = open(f"{test_save_path}", "w", encoding="utf8")

        # clean notes
        cleaned_test_notes = self.clean_data(
            admin_language=self.admin_language,
            remove_punctuation=self.remove_punctuation,
            replacement_map=self.replacement_map,
            notes_df=test_notes_temp,
        )

        # write text data alone for LM modelling etc
        cleaned_test_text = cleaned_test_notes[text_col].to_list()
        # write to file
        open_file.write("\n".join(cleaned_test_text))

        end = time.time()

        logger.warning(f"That took a total of: {end - start} seconds!")


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_test_notes_path", type=str, help="The path to the folder with data files"
                        )
    parser.add_argument(
        "--save_path",
        type=str,
        help="The directory to save processed and cleaned data files",
    )
    parser.add_argument(
        "--num_files_for_training",
        type=int,
        help="The number of files in your split data folder to use for training (the rest are used for test)",
    )

    parser.add_argument(
        "--admin_language",
        default=[
            "FINAL REPORT",
            "Date/Time",
            "Phone",
            "Date of Birth",
            "DoB",
            "Completed by",
            "Dictated by",
            "name of assessor:",
            "assessed by",
            "private and confidential",
            "\t",
        ],
        type=list,
        help=(
            "User defined list of strings to replace during cleaning using "
            "regular expression"
        ),
    )
    parser.add_argument(
        "--replacement_map",
        default=[
            ("\n", " "),
            ("\r\n\r", " "),
            ("\r", " "),
            ("\t", " "),
            ("w/", "with"),
            ("_", " "),
            ("*", " "),
            ("  ", " "),
            ('"', ""),
            ("-", " "),
        ],
        type=list,
        help=(
            "set of tuples to act as replacement maps. The first element of each "
            "tuple will be replaced by the second iteratively"
        ),
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether or not to process a sub sample of the data",
    )
    parser.add_argument(
        "--train_sample_size",
        default=120000,
        type=int,
        help="The sample size to use when subsetting training data",
    )
    parser.add_argument(
        "--test_sample_size",
        default=120000,
        type=int,
        help="The sample size to use when subsetting test data",
    )

    parser.add_argument(
        "--text_col",
        default="TEXT",
        type=str,
        help="The name of the column with the text data in",
    )

    args = parser.parse_args()

    print(f"vars args: {vars(args)}")

    # instantiate class with all arguments provided
    text_dataset = LMTextData(**vars(args))

    # text_dataset = LMTextData(
    #   brc_ids_path=args.brc_ids_path,
    #   notes_path=args.notes_path,
    #   training_save_path=args.training_save_path
    # )

    # now run the read_write_all_text function for training data
    text_dataset.read_write_all_text()


if __name__ == "__main__":
    main()
