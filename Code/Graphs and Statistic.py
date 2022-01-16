import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('normalized_sentiment_score.csv')

'#Generating descriptive statistics by subreddit'
x = df[["weighted_sentiment_score", "comment_subreddit"]].groupby("comment_subreddit").describe()
type(x)
print(x)

x.to_csv('summary_stats.csv')


'#Generating histogram for dataset'

plt.hist(df['weighted_sentiment_score'], bins = 5)
plt.xlabel('Weighted Sentiment Score')
plt.ylabel('Count')
plt.title('Reddit Community')
plt.savefig('Reddit Community.png')
plt.clf()



'#Generating a histogram for each subreddit'
unique_names = df.comment_subreddit.unique()

for x in unique_names:
    a = df[df['comment_subreddit'] == x]
    plt.hist(a['weighted_sentiment_score'], bins=5)
    plt.xlabel('Weighted Sentiment Score')
    plt.ylabel('Count')
    plt.title(x)
    plt.savefig(x + '.png')
    plt.clf()


