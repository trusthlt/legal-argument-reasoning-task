import fitz
import os
import re
from collections import defaultdict
from nltk import word_tokenize
import csv
import string


def read_pdf_into_chapters(
    path=os.path.join(os.getcwd(), "data", "raw", "book-legal.pdf")
):
    """Parse the pdf version of 'The Glannon Guide To Civil Procedure' to get the raw text.
    Separate the content into chapters

    Args:
        path (string, optional): Path to book pdf. Defaults to os.path.join(os.getcwd(), 'data', 'raw', 'book-legal.pdf').

    Returns:
        dict: Chapter separated text.
    """

    def clean_text(text):
        """
        normalize the text.
        """
        return text.replace("\t", " ").replace("  ", " ")

    chapter_dict = {}
    with fitz.open(path) as f:
        chapter_names = [
            c for c in f.get_toc() if c[0] == 1
        ]  # Remove first chapter subchapters which are unnecesary. Other subchapters are not detected
        for i in range(
            len(chapter_names) - 1
        ):  # Last chapter is index chapter and therefore not mandatory to collect
            chapter_content = ""
            start_page = chapter_names[i][2] - 1  # Page of chapter start
            end_page = chapter_names[i + 1][2]  # Page of chapter end

            for page in f.pages(start_page, end_page, 1):
                chapter_content += (
                    clean_text(page.getText()) + "\n"
                )  # Add page content to chapter content for later processing

            chapter_dict[
                chapter_names[i][1]
            ] = chapter_content  # Add chapter content to dictionary
    return chapter_dict


def write_into_csv(
    file_path,
    subchapter_list,
    title_list=["question", "answer", "solution", "analysis", "explanation"],
):
    """Write the parsed content in a dataset format into a csv file .

    Args:
        file_path (string): path to file
        subchapter_list (list): dataset entries
        title_list (list, optional): header. Defaults to ["question", "answer", "solution", "analysis", "explanation"].
    """
    with open(file_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter="\t", quotechar="|")
        writer.writerow(title_list)
        for entry in subchapter_list:
            writer.writerow(entry)


def get_title_name(raw_title_and_explanation, chars_per_line_length=61):
    """Get the title of the subchapter by evaluating the char length per line.
    The text is written "Blocksatz" formatting stratching the words per line to fit the whole line.
    Comparing the char length would allow to differ between the bigger written subchapter header and the smaller written explanation.
    Furthermore could the title be treated as the first paragraph of the subchapter.

    Args:
        raw_title_and_explanation (string): raw text of the subchapter

    Returns:
        title (string): title of the subchapter
        explanation (string): explanation of the subchapter
    """
    title = ""
    explanation = ""
    first_explanation_line = -1
    raw_title_and_explanation_list = raw_title_and_explanation.split("\n")
    for i in range(len(raw_title_and_explanation_list)):
        if len(raw_title_and_explanation_list[i]) < chars_per_line_length:
            title += (
                raw_title_and_explanation_list[i] + " "
            )  # First paragraph is title of subchapter

        else:
            first_explanation_line = i  # + 1
            break

    explanation = " ".join(raw_title_and_explanation_list[first_explanation_line:])
    return title, explanation


def split_char_dot_pattern(raw):
    """
    Split the raw text at the recurring keyword pattern "<Character>."
    """
    return re.split(
        r"\n[A-H]\. ", raw
    )  # A letter after H is not occuring in the text as Headliner but as Normal letter.


def process_chapter(raw_chapter, chapter_index):
    """Divide the chapter into its subchapters. Moreover divide and annotate each subchapter into the 4 parts: explanation, question, answers, analysis

    Args:
        raw_chapter (string): raw text
        chapter_index (int): Index of the chapter

    Returns:
        dict: dict of lists. Each subchapter is an key with a list containing the separated parts: explanation, question, answers, analysis.
    """

    def fallback():
        return {}  # defaultdict fallback option

    def parse_chapter_solutions(raw_solutions, number_of_questions):
        """
        Parse the solutions subchapter and put the solution character in a list.
        Transforms the character into a number (0,1).
        """
        solutions = []
        for solution in re.split(
            r"\n[0-9]+\. ",
            raw_solutions,
        )[1:number_of_questions]:
            solution_char = [
                sol
                for sol in word_tokenize(solution)
                if len(sol) == 1 and sol in string.ascii_uppercase
            ][-1]  # Last Token in split is the solution character; Hacky solution but book is not consistent with annotations
            solution_number = [
                index
                for index, char in enumerate(string.ascii_uppercase)
                if char == solution_char
            ][0]  # Transform the character into a number to make parsing easier
            solutions.append(solution_number)

        return solutions

    def add_to_dict(question, answer, analysis):
        """Add new entry to output dict

        Args:
            question (string): parsed question
            answer (string): parsed answer
            analysis (string): parsed analysis
        """
        if (
            "question" in chapter_dict[last_chapter_name].keys()
        ):  # If multiple questons are in the same chapter append them
            chapter_dict[last_chapter_name]["question"].append(question)
            chapter_dict[last_chapter_name]["answers"].append(answer)
            chapter_dict[last_chapter_name]["analysis"].append(analysis)
        else:
            chapter_dict[last_chapter_name]["question"] = [question]
            chapter_dict[last_chapter_name]["answers"] = [answer]
            chapter_dict[last_chapter_name]["analysis"] = [analysis]

    def special_rules():
        """Lookup the correct keyword pattern if analysis split is not available. keyword pattern were added manually through looking it up in the book

        Returns:
            string: keyword pattern
        """
        if chapter_index == 5:
            keyword_pattern = "There is a significant difference here that would likely lead to a different\n\noutcome."
        elif chapter_index == 9:
            keyword_pattern = "Let’s start in the middle and nibble up and down. "
        else:
            print("Index not in list with special rules. Exit")
            exit()
        return keyword_pattern

    last_chapter_name = ""
    chapter_dict = defaultdict(fallback)
    solutions = []  # store the solutions of the questions
    chapter_order = (
        []
    )  # Store chapter names in order to assing the correct solution (getting at the end of the chapter) to the correct question
    for section in re.split(
        r"QUESTION\d*.", raw_chapter
    ):  # Divide chapter into sections between the stable keyword pattern "Question <Number>."
        if last_chapter_name == "":  # First chapter border case
            title_split = split_char_dot_pattern(section)[
                -1
            ]  # Last section is the title and explanation. The first sections is Chapter overview
            new_chapter_name, explanation = get_title_name(title_split)
            chapter_dict[new_chapter_name]["explanation"] = explanation

        elif "Glannon’s Picks" in section:  # Last chapter border case
            analysis_split = re.split(r"\n*ANALYSIS\. ", section, re.MULTILINE)

            if len(analysis_split) == 1:
                analysis_split = re.split(
                    "\n*%s" % (special_rules()), section, re.MULTILINE
                )

            subchapter_question_and_answer_split = split_char_dot_pattern(
                analysis_split[0]
            )  # split question and question answers from next chapter title and explanation
            analysis_solution_split = re.split(
                r"Glannon’s Picks", " ".join(analysis_split[1:]), re.MULTILINE
            )  # Split at Keyword Glannon's Picks
            add_to_dict(
                subchapter_question_and_answer_split[0],
                subchapter_question_and_answer_split[1:],
                analysis_solution_split[0],
            )
            solutions = parse_chapter_solutions(
                analysis_solution_split[1], len(chapter_order) + 1
            )  # Part after subchapter title

        else:  # Normal/Middle chapter case
            print("main case start")
            analysis_split = re.split(r"\n*ANALYSIS\. ", section, re.MULTILINE)

            if len(analysis_split) == 1:
                analysis_split = re.split(
                    "\n*%s" % (special_rules()), section, re.MULTILINE
                )

            subchapter_question_and_answer_split = split_char_dot_pattern(
                analysis_split[0]
            )  # split question and question answers from next chapter title and explanation
            subchapter_title_and_explanation_split = split_char_dot_pattern(
                analysis_split[1]
            )  # split question and analysis from next chapter title and explanation

            if (
                len(subchapter_title_and_explanation_split) == 1
            ):  # No next chapter title and explanation
                add_to_dict(
                    subchapter_question_and_answer_split[0],
                    subchapter_question_and_answer_split[1:],
                    analysis_split[1],
                )
                chapter_order.append(new_chapter_name)
                continue  # No new chapter therfore no new name assignments
            else:
                new_chapter_name, explanation = get_title_name(
                    subchapter_title_and_explanation_split[-1]
                )
                add_to_dict(
                    subchapter_question_and_answer_split[0],
                    subchapter_question_and_answer_split[1:],
                    " ".join(subchapter_title_and_explanation_split[0:-1]),
                )
                chapter_dict[new_chapter_name]["explanation"] = explanation

        last_chapter_name = new_chapter_name
        chapter_order.append(new_chapter_name)

    for chapter_name, solution in zip(
        chapter_order, solutions
    ):  # Add solutions to the chapter_dict
        if "solution" not in chapter_dict[chapter_name].keys():
            chapter_dict[chapter_name]["solution"] = [solution]
        else:
            chapter_dict[chapter_name]["solution"].append(solution)

    return chapter_dict


def parse_chapter_26(raw_chapter):
    """Divide the special chapter 26 into its subchapters. Moreover divide and annotate each subchapter into the 4 parts: explanation, question, answers, analysis

    Args:
        raw_chapter (string): raw text

    Returns:
        dict: dict of lists. Each subchapter is an key with a list containing the separated parts: explanation, question, answers, analysis.
    """
    chapter_dict = {}
    questions, solutions = re.split(r"Glannon’s Picks", raw_chapter, re.MULTILINE)

    for index, section in enumerate(
        re.split(r"QUESTION\d*.", questions)[1:]
    ):  # Ignore chapter introduction
        question_and_answers_split = split_char_dot_pattern(section)
        chapter_dict[index] = {
            "question": [question_and_answers_split[0]],
            "answers": [question_and_answers_split[1:]],
            "solution": [-1],
            "explanation": "",
        }

    for index, section in enumerate(re.split(r"\n\d{1,2}\. ", solutions)[1:17]):
        if index == 16:
            chapter_dict[index]["analysis"] = [section[:-3]]  # Ignore last three lines
        else:
            chapter_dict[index]["analysis"] = [section]

    return chapter_dict


def generate_ds_pairs(chapter_dict):
    """Reformat the parsed content from a multiple choice question type into a binary question type

    Args:
        chapter_dict (list): chapter content divided into its parts

    Returns:
        list: list of dataset entries
    """

    def clean_text(text):
        return text.replace("\n", " ").replace("  ", " ")

    ds_entries = []
    for subchapter in chapter_dict.keys():
        for question, answers, analysis, solution in zip(
            chapter_dict[subchapter]["question"],
            chapter_dict[subchapter]["answers"],
            chapter_dict[subchapter]["analysis"],
            chapter_dict[subchapter]["solution"],
        ):
            for index, answer in enumerate(answers):
                answer_solution = 1 if index == solution else 0
                answer = answer.replace("\n", " ").replace("  ", " ")  # Cleaning
                ds_entries.append(
                    [
                        clean_text(question),
                        clean_text(answer),
                        answer_solution,
                        clean_text(analysis),
                        clean_text(chapter_dict[subchapter]["explanation"]),
                    ]
                )

    return ds_entries


if __name__ == "__main__":
    chapter_dict = read_pdf_into_chapters()
    for index, key in enumerate(chapter_dict.keys()):
        if index in [
            0,
            27,
        ]:  # Chapter 0 and 27 can be ignored as they do not entries for the final dataset
            continue
        if (
            index == 26
        ):  # Chapter 26 does not follows the structure of the other chapters.
            chapter = parse_chapter_26(chapter_dict[key])
        else:
            chapter = process_chapter(chapter_dict[key], index)

        ds_pairs = generate_ds_pairs(chapter)
        write_into_csv(
            os.path.join(os.getcwd(), "data", "processed", key + "_ds_pairs.csv"),
            ds_pairs,
        )
