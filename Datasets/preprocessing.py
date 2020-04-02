import os
import pandas as pd


def content_process(content, topic):
    # change capital letters to lower case letters
    new_content = content.lower()
    temp = topic.lower()

    # remove special symbols
    for a in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n':
        new_content = new_content.replace(a, '')
        temp = temp.replace(a, '')

    # replace URL as 'URL'
    while new_content.find('http') != -1:
        start = new_content.find('http')
        end = start + 3
        while end + 1 < len(new_content):
            if new_content[end + 1] == ' ':
                break
            else:
                end = end + 1
        new_content = new_content.replace(new_content[start: end+1], 'URL')

    # if topic is not in content, add it in front of content
    if new_content.find(temp) == -1:
        new_content = temp + ' ' + new_content
    return new_content


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
                train_df['content'].append(content_process(new_df['content'][i], new_df['topic'][i]))
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
        test_df['content'].append(content_process(new_df['content'][i], new_df['topic'][i]))
        test_df['label'].append(label)

test_df = pd.DataFrame(test_df)
test_df.to_csv('./my_test_data.csv', index=False)