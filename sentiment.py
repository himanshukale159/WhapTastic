import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def sentiment_analysis(df):
        
    sentiment = SentimentIntensityAnalyzer()

    df["positive"] = [sentiment.polarity_scores(i)["pos"] for i in df['message']]
    df['negative'] = [sentiment.polarity_scores(i)["neg"] for i in df['message']]
    df['neutral'] = [sentiment.polarity_scores(i)["neu"] for i in df['message']]
    user_score = df.groupby('user')[['positive', 'negative', 'neutral']].mean().reset_index()
    return user_score

def plot_sentiment(selected_user,user_score):

    if selected_user == 'Overall':
        mean_positive = user_score['positive'].mean()
        mean_negative = user_score['negative'].mean()
        mean_neutral = user_score['neutral'].mean()

        labels = ['Positive','Negative','Neutral']
        sizes = [mean_positive,mean_negative,mean_neutral]

    else:
        user_row = user_score.loc[user_score['user'] == selected_user]
        positive_val = user_row['positive'].values[0]
        negative_val = user_row['negative'].values[0]
        neutral_val = user_row['neutral'].values[0]

        labels = ['Positive','Negative','Neutral']
        sizes = [positive_val,negative_val,neutral_val]

    return labels,sizes

        