import time

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB

from utils.logger import logger


def label_your_texts(texts: list[str], test_size: float = 0.3):
    start_time = time.perf_counter()
    logger.info(
        'Started using our trained model to generate labels for blogs.'
    )

    newsgroups_data = fetch_20newsgroups(
        subset='all', shuffle=True, random_state=42
    )

    X = newsgroups_data.data
    y = newsgroups_data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    model = ComplementNB()
    model.fit(X_train_vectorized, y_train)

    accuracy = model.score(X_test_vectorized, y_test)

    logger.debug(
        f"Model with \u001b[32m{accuracy * 100:.2f} % \u001b[37m"
        f"accuracy will be used."
    )

    vectorized_texts = vectorizer.transform(texts)
    topic_names = newsgroups_data.target_names
    generated_labels = [
        topic_names[label] for label in model.predict(vectorized_texts)
    ]
    logger.debug(
        f'Successfully generated labels for blogs. Time Taken: \u001b[32m'
        f'{(time.perf_counter() - start_time):.2f} seconds\u001b[37m.'
    )
    return generated_labels


if __name__ == '__main__':
    test_texts = ["""
    My Fellow Americans,

    I stand before you today not to spew fancy political rhetoric or to talk 
    about what could be better. I’m here today because you all are here. And 
    you’re clearly fed up with politics. I’m here because our democracy isn’t 
    working for all of us. You all know that this is the most important 
    election 
    of our lifetime. So I’m going to do something. This is why I, John Smith, 
    am running to be the Town Council of District 40.

    You may think more divides us than unites us in this new era of politics. 
    But 
    I know we can all unite behind Small Business. Not only do our lives depend 
    on making progress on Small Business, but so do the lives of our children.

    However, putting our all towards Small Business won’t be enough to save our 
    democracy. I know each and every single one you is struggling. That’s why I 
    am also going to also focus on Taxes and Free College for All. The elites 
    in 
    power feel too comfortable, so we need to shake things up. I am here 
    today to 
    listen to you, the American people, because I know you have been silenced 
    for 
    far too long.

    Vote John Smith for Town Council of District 40!
    """]
    label_your_texts(texts=test_texts)
