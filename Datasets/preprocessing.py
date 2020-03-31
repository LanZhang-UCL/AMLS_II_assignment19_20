import os
import pandas as pd

# Process raw training data and generate new training data.
for root, _, files in os.walk('./Subtasks_BD/'):
    train_df = {'id':[], 'topic':[], 'label':[], 'content':[]}
    for file in files:
        new_df = pd.read_csv(root+file,
                 sep='\t',
                 index_col=False,
                 names=['id', 'topic', 'label', 'content'],
                 dtype=str)
        for i in range(0, len(new_df['id'])):
            if new_df['label'][i] != 'neutral':
                if new_df['label'][i] == 'positive':
                    label = 1
                else:
                    label = 0
                train_df['id'].append(new_df['id'][i])
                train_df['topic'].append(new_df['topic'][i])
                content = new_df['topic'][i] + ' ' + new_df['content'][i]
                train_df['content'].append(content.lower())
                train_df['label'].append(label)

train_df = pd.DataFrame(train_df)
train_df.to_csv('./my_training_data.csv', index=False)

# Process raw test data and generate new test data.
test_df = {'id':[], 'topic':[], 'label':[], 'content':[]}
new_df = pd.read_csv('./SemEval2017-task4-test.subtask-BD.english.txt',
                 sep='\t',
                 index_col=False,
                 names=['id', 'topic', 'label', 'content'],
                 dtype=str)
for i in range(0, len(new_df['id'])):
    if new_df['label'][i] != 'neutral':
        if new_df['label'][i] == 'positive':
            label = 1
        else:
            label = 0
        test_df['id'].append(new_df['id'][i])
        test_df['topic'].append(new_df['topic'][i])
        content = new_df['topic'][i] + ' ' + new_df['content'][i]
        test_df['content'].append(content.lower())
        test_df['label'].append(label)

test_df = pd.DataFrame(test_df)
test_df.to_csv('./my_test_data.csv', index=False)