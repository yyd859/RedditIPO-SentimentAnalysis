from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import ftfy
'#building model'
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

'#importing the dataset'
df = pd.read_csv('comments_rd.csv')
print(df.head())
data=df['comment_body']
print('successfully imported data')

'#attempt to fix garrble'
df['fixed_body'] = df['comment_body'].apply(lambda x: ftfy.fix_text(x))
print('successfully fixed text')

'#building function'
def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1

def sentiment_score_adjusted(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))-2

print('generating sentiment')
df['sentiment'] = df['fixed_body'].apply(lambda x: sentiment_score(x[:512]))
print('generated sentiment score')

print('generating adjusted sentiment')
df['sentiment_adjusted'] = df['fixed_body'].apply(lambda x: sentiment_score_adjusted(x[:512]))
print('generated adjusted sentiment score')

df.to_csv('sentiment_adjusted_v2.csv')

print(df)