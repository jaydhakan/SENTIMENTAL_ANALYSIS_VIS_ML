import matplotlib.pyplot as plt
import numpy as np


def generate_visualisation(
    positive_scores: list, negative_scores: list, topic_names: list
):
    total_positive_scores = {}
    total_negative_scores = {}
    total_blogs_on_topic = {}
    for topic, positive_score, negative_score in zip(
        topic_names, positive_scores, negative_scores
    ):
        if topic in total_blogs_on_topic.keys():
            total_blogs_on_topic[topic] += 1
        else:
            total_blogs_on_topic[topic] = 1
        total_positive_scores[topic] = int(
            (total_positive_scores.get(topic, 0)) + positive_score
        )

        total_negative_scores[topic] = int(
            (total_negative_scores.get(topic, 0)) + negative_score
        )

    topic_names = total_positive_scores.keys()

    fig, ax = plt.subplots(figsize=(15, 15))

    y = range(len(topic_names))
    height = 0.35

    bars1 = ax.barh(
        y, total_positive_scores.values(),
        height, label='Positive Score', color='#005E63'
    )
    bars2 = ax.barh(
        y, total_negative_scores.values(), height, color='darkorange',
        label='Negative Score', left=list(total_positive_scores.values())
    )

    ax.set_ylabel('Topic Category (blog count)', fontweight='bold')
    ax.set_xlabel('Scores', fontweight='bold')
    ax.set_title('Positive and Negative Scores by Topic Category')
    ax.set_yticks(y)
    y_ticks_labels = [
        f'{topic} ({total_blogs_on_topic[topic]})' for topic in topic_names
    ]
    ax.set_yticklabels(y_ticks_labels)

    for bar1, bar2 in zip(bars1, bars2):
        ax.text(
            bar1.get_width() / 2, bar1.get_y() + bar1.get_height() / 2,
            f'{int(bar1.get_width())}', ha='left', va='center',
            color='white', fontweight='bold', fontsize=8
        )
        ax.text(
            bar1.get_width() + (bar2.get_width()) / 2 + 1,
            bar2.get_y() + bar2.get_height() / 2,
            f'{int(bar2.get_width())}', ha='left', va='center',
            color='black', fontweight='bold', fontsize=8
        )

    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname='positive_negative_scores_by_topic')


def generate_line_graph(positive_scores: list, negative_scores: list):
    x = range(1, len(positive_scores) + 1)
    plt.plot(x, positive_scores, color='green', label='Positive Words')
    plt.plot(x, negative_scores, color='#FF474C', label='Negative Words')

    plt.xlabel('Blogs')
    plt.ylabel('Scores')
    plt.title('Blogs vs Positive and Negative Scores')
    plt.legend()
    plt.grid(True)

    plt.savefig('positive_negative_scores_by_urls')


def generate_complex_words_graph(
    positive_scores: list, negative_scores: list, complex_words_count: list
):
    sentiment_difference = np.array(positive_scores) - np.array(negative_scores)
    fig, ax = plt.subplots()
    plt.scatter(complex_words_count, sentiment_difference)
    ax.set_title('Impact of Complex Words on Sentiment of blog')
    ax.set_xlabel('Frequency of Complex Words')
    ax.set_ylabel('Difference between Positive and Negative Words')
    plt.grid(True)

    plt.savefig('complex_words-vs-sentiment')
