import re
import time
from os import listdir, path, walk

import requests
from bs4 import BeautifulSoup as Soup
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from utils.logger import logger


class AnalysisHelper:
    cleaned_blogs = None
    start_time = time.perf_counter()
    file_paths = None

    def write_responses_from_blogs_in_files(
        self, dataframe, no_of_url_to_scrap: int = 100
    ):
        start_time = time.perf_counter()
        urls = [url for url in dataframe['URL']][:no_of_url_to_scrap]
        url_ids = [url_id for url_id in dataframe['URL_ID']][
                  :no_of_url_to_scrap]

        logger.info(
            f'Scraping started on \u001b[32m{len(urls)}\u001b[37m urls.'
        )

        response_list = [
            requests.get(url, headers={"User-Agent": "XY"}) for url in urls
        ]

        response_list = [
            Soup(response.content, 'html.parser') for response in response_list
        ]

        corrupted_url_ids = []
        for url_id, response in zip(url_ids, response_list):
            try:
                response_text = response.find(
                    attrs={"class": "td-post-content"}
                ).text
                with (
                    open(f'blogs/{url_id}.txt', 'w', encoding='utf-8')
                    as response_file
                ):
                    response_text = response_text.replace('\n', '')
                    response_file.write(response_text)
            except Exception as error:
                corrupted_url_ids.append(url_id)
                logger.error(
                    f'Could not parse html for site number: {url_id}, '
                    f'ERROR: {error}'
                )

        self.file_paths = [
            path.join(root, file) for root, _, files in walk('blogs')
            for file in files
        ]
        logger.debug(
            f'Successfully scraped {len(self.file_paths)} urls. Time taken: '
            f'\u001b[32m{(time.perf_counter() - start_time):.2f} seconds.'
            f'\u001b[37m'
        )
        logger.info(
            f'Analysis will be done on {len(self.file_paths)} blogs.'
        )
        filtered_df = dataframe[~dataframe['URL_ID'].isin(corrupted_url_ids)]
        return filtered_df

    def generate_file_paths_of_scrapped_blogs(self):
        if self.file_paths is None:
            self.file_paths = [
                path.join(root, file) for root, _, files in walk('blogs')
                for file in files
            ]
            logger.info(
                f'Analysis will be done on '
                f'\u001b[32m{len(self.file_paths)}\u001b[37m blogs.'
            )

    def read_text_from_files(self):
        self.generate_file_paths_of_scrapped_blogs()
        for file_path in self.file_paths:
            with open(file_path, 'r', encoding='utf-8') as text_file:
                yield text_file.read()

    def get_sentences(self):
        self.generate_file_paths_of_scrapped_blogs()
        sentences = []
        for file_path in self.file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                sentence_count = len(sent_tokenize(file.read()))
                sentences.append(sentence_count)
        return sentences

    @staticmethod
    def get_list_of_stop_words():
        stop_words = list(set(stopwords.words('english')))
        for file in listdir('details/StopWords'):
            if file.endswith('.txt'):
                with open(
                    path.join('details/StopWords', file), 'r',
                    encoding='latin-1'
                ) as f:
                    stop_words.extend(f.readlines())

        cleaned_stop_words = []
        for stop_word in stop_words:
            words = stop_word.split(' | ')
            words = [
                words.pop(0).strip() if 'http' in word
                else word.replace('\n', '').strip() for word in words
            ]
            cleaned_stop_words.extend(words)
        for stop_word in cleaned_stop_words:
            if len(stop_word) < 2:
                cleaned_stop_words.remove(stop_word)
        return list(set(cleaned_stop_words))

    def get_cleaned_blogs(self):
        pattern = '|'.join(map(re.escape, ['?', '.', ',', '!']))
        cleaned_blogs = [re.sub(pattern, ' ', blog)
                         for blog in list(self.read_text_from_files())]

        characters_to_replace = [
            f' {word} ' for word in self.get_list_of_stop_words()
        ]
        pattern = '|'.join(map(re.escape, characters_to_replace))
        self.cleaned_blogs = [
            re.sub(pattern, ' ', blog) for blog in cleaned_blogs
        ]

    def get_words_count_from_blogs(self):
        self.get_cleaned_blogs()
        words_count = [len(word_tokenize(blog))
                       for blog in list(self.read_text_from_files())]

        cleaned_words_count = [
            len(word_tokenize(blog)) for blog in self.cleaned_blogs
        ]
        return words_count, cleaned_words_count

    def get_positive_and_negative_score(self):
        positive_words = set()
        with open(
            'details/MasterDictionary/positive-words.txt', 'r',
            encoding='utf-8'
        ) as words_file:
            positive_words.update(words_file.readlines())
            positive_words = {word.replace('\n', '') for word in positive_words}

        negative_words = set()
        with open(
            'details/MasterDictionary/negative-words.txt', 'r',
            encoding='latin-1'
        ) as words_file:
            negative_words.update(words_file.readlines())
            negative_words = {word.replace('\n', '') for word in negative_words}

        positive_score = [
            sum(1 for word in blog.lower().split(' ') if word in positive_words)
            for blog in self.cleaned_blogs
        ]
        negative_score = [
            sum(1 for word in blog.lower().split(' ') if word in negative_words)
            for blog in self.cleaned_blogs
        ]
        logger.debug(f'Successfully calculated positive and negative score.')
        return positive_score, negative_score

    @staticmethod
    def count_syllables(words_list: list):
        syllables_count = 0
        for word in words_list:
            if word[-2:] in ['es', 'ed']:
                word = word[:-2]
            vowels = r'[aeiouAEIOU]'
            syllables_count += len(re.findall(vowels, word))
        return syllables_count

    def count_complex_words(self, words_list: list):
        complex_words = 0
        for word in words_list:
            if len(word) > 2 and self.count_syllables(word) > 2:
                complex_words += 1
        return complex_words

    @staticmethod
    def count_personal_pronouns(text):
        pronoun_list = ["I", "we", "my", "ours", "us", "We", "My", "Ours", "Us"]
        pronoun_pattern = '|'.join(map(re.escape, pronoun_list))
        pronoun_count = len(re.findall(pronoun_pattern, text))
        return pronoun_count

    def get_all_counts_for_analysis(self):
        syllable_counts = []
        complex_words_counts = []
        personal_pronouns_counts = []
        for text in self.cleaned_blogs:
            words_list = text.split()
            syllable_counts.append(self.count_syllables(words_list))
            complex_words_counts.append(self.count_complex_words(words_list))
            personal_pronouns_counts.append(self.count_personal_pronouns(text))
        logger.debug(
            f'Successfully calculated syllable_counts, complex_words_counts '
            f'and personal_pronouns_counts.'
        )
        return syllable_counts, complex_words_counts, personal_pronouns_counts

    def get_total_characters(self):
        total_characters = []
        for blog in list(self.read_text_from_files()):
            characters = 0
            for word in blog.split():
                characters += len(word)
            total_characters.append(characters)
        logger.debug(
            f'Successfully calculated all the things. Time taken: \u001b[32m'
            f'{(time.perf_counter() - self.start_time):.2f} '
            f'seconds.\u001b[37m\n'
        )
        return total_characters


analysis_helper = AnalysisHelper()
if __name__ == '__main__':
    pass
