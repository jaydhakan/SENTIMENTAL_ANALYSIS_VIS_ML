import os
import time

import numpy as np
import pandas as pd

from helpers import (
    get_all_counts_for_analysis,
    get_positive_and_negative_score,
    get_sentences, get_set_of_stop_words,
    get_total_characters, get_words_count_from_blogs,
    remove_the_stop_words_from_blogs,
    write_responses_from_blogs_in_files
)
from label import label_your_texts
from visualisation_helpers import generate_line_graph, generate_visualisation


def start_analysis(is_data_extracted: bool = False):
    df = pd.read_excel('assignment_details/Input.xlsx')
    # TODO: uncomment this
    # url_ids = [url_id[11:] for url_id in df['URL_ID']]
    # df['URL_ID'] = url_ids

    start_time = time.perf_counter()
    if not is_data_extracted:
        df = write_responses_from_blogs_in_files(df)

    df = df[~df['URL_ID'].isin(['blackassign0036', 'blackassign0049'])]
    file_paths = [
        os.path.join(root, file) for root, _, files in os.walk('blogs')
        for file in files
    ]
    print(f'Analysis will be done on {len(file_paths)} urls.\n')

    stop_words = get_set_of_stop_words()
    print(
        f'Total {len(stop_words)} stop words found '
        f'in our dictionary after cleaning.\n'
    )

    sentences = get_sentences(file_paths)

    def read_text_from_files():
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as text_file:
                yield text_file.read()

    cleaned_blogs = remove_the_stop_words_from_blogs(
        stop_words, list(read_text_from_files())
    )
    print(
        f'Successfully removed the stop words. '
        f'Time taken: {time.perf_counter() - start_time}\n'
    )

    words_count, cleaned_words_count = get_words_count_from_blogs(
        list(read_text_from_files()), cleaned_blogs
    )
    cleaned_words_count = np.array(cleaned_words_count)

    positive_score, negative_score = get_positive_and_negative_score(
        cleaned_blogs
    )
    print(f'Successfully calculated positive and negative score.\n')

    syllable_counts, complex_words_counts, personal_pronouns_counts = (
        get_all_counts_for_analysis(cleaned_blogs)
    )
    print(
        f'Successfully calculated syllable_counts, complex_words_counts and '
        f'personal_pronouns_counts.\n'
    )

    total_characters = get_total_characters(list(read_text_from_files()))
    print(f'Successfully calculated all the things.\n')

    df['POSITIVE SCORE'] = positive_score
    df['NEGATIVE SCORE'] = negative_score

    df['POLARITY SCORE'] = (
        (df['POSITIVE SCORE'] - df['NEGATIVE SCORE']) /
        ((df['POSITIVE SCORE'] + df['NEGATIVE SCORE']) + 0.000001)
    )

    df['SUBJECTIVITY SCORE'] = (
        (df['POSITIVE SCORE'] + df['NEGATIVE SCORE']) /
        (cleaned_words_count + 0.000001)
    )

    df['PERCENTAGE OF COMPLEX WORDS'] = (
        np.array(complex_words_counts) / np.array(words_count)
    )

    df['AVG NUMBER OF WORDS PER SENTENCES'] = (
        np.array(words_count) / np.array(sentences)
    )

    df['FOG INDEX'] = 0.4 * (
        df['AVG NUMBER OF WORDS PER SENTENCES'] +
        df['PERCENTAGE OF COMPLEX WORDS']
    )

    df['COMPLEX WORD COUNT'] = complex_words_counts
    df['WORD COUNT'] = words_count
    df['SYLLABLE PER WORD'] = np.array(syllable_counts) / np.array(words_count)
    df['PERSONAL PRONOUN'] = personal_pronouns_counts
    df['AVG WORD LENGTH'] = np.array(total_characters) / np.array(words_count)

    print('Started using our trained model to generate labels for blogs.\n')

    start_time = time.perf_counter()
    labels_of_text = label_your_texts(texts=list(read_text_from_files()))
    df['TOPIC'] = np.array(labels_of_text)
    print(
        'Successfully generated labels for blogs. '
        f'Time Taken: {time.perf_counter() - start_time}\n'
    )

    df.to_excel('Output.xlsx', index=False)
    print(f'Successfully created output excel file.\n')

    print('Generating visualisations using matplotlib library.')
    generate_visualisation(positive_score, negative_score, labels_of_text)
    generate_line_graph(positive_score, negative_score)


if __name__ == '__main__':
    start_analysis(is_data_extracted=True)

'''
Word Count vs. Positive/Negative Score: Investigate the relationship between 
the length of texts (word count) and their sentiment scores. You could create 
scatter plots or box plots to visualize this relationship.

Percentage of Complex Words vs. Sentiment: Explore whether there's a 
relationship between the percentage of complex words in a text and its 
sentiment scores. You could create scatter plots or box plots to examine this 
relationship.
'''
