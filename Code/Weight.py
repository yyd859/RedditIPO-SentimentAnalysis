import pandas as pd

'#importing the dataset'
df = pd.read_csv('sentiment_adjusted_v2.csv')
print(df)

'#grabbing the count'
def subcomment_count(df):
    a=df.comment_parent_id.str.split(pat="_",n=2,expand=True)
    a=a.iloc[:,[1]]
    b=a.apply(lambda x : x.groupby(x).count()).fillna(0)
    b.index.name = 'comment_id'
    b.reset_index(inplace=True)
    b.rename(columns={1:'subcomment_count'}, inplace=True)
    c=df.merge(b, on='comment_id', how='left')
    return c

'#creating weight'
count = subcomment_count(df)
count['subcomment_count'] = count['subcomment_count'].fillna(0)

print(count)

'#normalizing weight'
count['subcomment_weight']=(count['subcomment_count'] - count['subcomment_count'].min()) / (count['subcomment_count'].max() - count['subcomment_count'].min())
count['upvote_weight'] = (count['comment_score'] - count['comment_score'].min()) / (count['comment_score'].max() - count['comment_score'].min())

print(count)

'#weighted sentiment score'
count['weighted_sentiment_score'] = count['sentiment_adjusted']*(1+count['upvote_weight'])*(1+count['subcomment_weight'])
count.to_csv('normalized_sentiment_score.csv')