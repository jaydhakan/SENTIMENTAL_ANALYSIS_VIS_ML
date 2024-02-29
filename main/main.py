import time

import numpy as np
import pandas as pd

from utils.analysis_helpers import analysis_helper
from utils.logger import logger
from utils.model import label_your_texts
from utils.visualisation_helpers import (
    generate_complex_words_graph, generate_line_graph, generate_visualisation
)


def start_analysis(is_data_extracted: bool = False):
    df = pd.read_excel('details/Input.xlsx')
    start_time = time.perf_counter()
    if not is_data_extracted:
        df = analysis_helper.write_responses_from_blogs_in_files(df)

    corrupted_url_ids = ['36', '49']
    df = df[~df['URL_ID'].isin(corrupted_url_ids)]

    sentences = analysis_helper.get_sentences()

    words_count, cleaned_words_count = (
        analysis_helper.get_words_count_from_blogs()
    )
    positive_score, negative_score = (
        analysis_helper.get_positive_and_negative_score()
    )

    syllable_counts, complex_words_counts, personal_pronouns_counts = (
        analysis_helper.get_all_counts_for_analysis()
    )

    total_characters = analysis_helper.get_total_characters()

    df['POSITIVE SCORE'] = positive_score
    df['NEGATIVE SCORE'] = negative_score

    df['POLARITY SCORE'] = (
            (df['POSITIVE SCORE'] - df['NEGATIVE SCORE']) /
            ((df['POSITIVE SCORE'] + df['NEGATIVE SCORE']) + 0.000001)
    )

    df['SUBJECTIVITY SCORE'] = (
            (df['POSITIVE SCORE'] + df['NEGATIVE SCORE']) /
            (np.array(cleaned_words_count) + 0.000001)
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

    labels_for_texts = label_your_texts(
        texts=list(analysis_helper.read_text_from_files())
    )
    df['TOPIC'] = np.array(labels_for_texts)

    df.to_excel('Output.xlsx', index=False)
    logger.debug(f'Successfully created output excel file.\n')

    logger.info('Generating visualisations using matplotlib library.')
    generate_line_graph(positive_score, negative_score)
    generate_visualisation(positive_score, negative_score, labels_for_texts)
    generate_complex_words_graph(
        positive_score, negative_score, complex_words_counts
    )

    logger.debug('Visualisations created.')

    logger.info(
        f'Total time taken: '
        f'\u001b[32m{(time.perf_counter() - start_time):.2f} seconds.\u001b[37m'
    )


if __name__ == '__main__':
    start_analysis(is_data_extracted=False)
