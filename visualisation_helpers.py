import matplotlib.pyplot as plt

_TRAIN_DOWNLOAD_URL = ("https://raw.githubusercontent.com/mhjabreel"
                       "/CharCnn_Keras/master/data/ag_news_csv/train.csv")
_TEST_DOWNLOAD_URL = ("https://raw.githubusercontent.com/mhjabreel"
                      "/CharCnn_Keras/master/data/ag_news_csv/test.csv")


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
            total_positive_scores.get(topic, 0) + positive_score
        )

        total_negative_scores[topic] = int(
            total_negative_scores.get(topic, 0) + negative_score
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

    ax.set_ylabel('Topic Category', fontweight='bold')
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
            color='white', fontweight='bold', fontsize=10
        )
        ax.text(
            bar1.get_width() + (bar2.get_width()) / 2 + 1,
            bar2.get_y() + bar2.get_height() / 2,
            f'{int(bar2.get_width())}', ha='left', va='center',
            color='black', fontweight='bold', fontsize=10
        )

    ax.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(fname='temp')


def generate_line_graph(positive_scores: list, negative_scores: list):
    fig, ax = plt.subplots()
    x = range(1, len(positive_scores) + 1)
    plt.plot(x, positive_scores, color='green', label='Positive Words')
    plt.plot(x, negative_scores, color='red', label='Negative Words')

    plt.xlabel('Data Points')
    plt.ylabel('Scores')
    plt.title('Positive and Negative Scores')
    plt.legend()

    plt.savefig('tmep2')
