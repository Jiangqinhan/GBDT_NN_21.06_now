import pandas as pd
import numpy as np
from ast import literal_eval
import os
from scipy.spatial import distance
import pickle

RootPath = r'D:\Great_job_of_teammate'
user_action_path = os.path.join(RootPath, "wechat_algo_data1/user_action.csv")
feed_info_modified_path = os.path.join(RootPath, "feed_info_modified2.pkl")
test_data_path = os.path.join(RootPath, "wechat_algo_data1/test_a.csv")
user_interest_path = os.path.join(RootPath, "user_interest.pkl")
LABEL_COLUMNS = ['click_avatar', 'forward', 'follow', 'favorite', 'read_comment', 'comment', 'like']
ACTION_LIST = ['click_avatar', 'forward', 'follow', 'favorite', 'read_comment', 'comment', 'like', 'is_stay',
               'is_finished', 'feedback', 'interested']
with open(feed_info_modified_path, 'rb') as file:
    feed_info_modified_df = pickle.load(file)

with open(user_interest_path, 'rb') as file:
    user_interest_df = pickle.load(file)
# 读取用户历史行为数据
user_action_df = pd.read_csv(user_action_path)
# 读取测试集数据
test_data_df = pd.read_csv(test_data_path)
# 统计视频第一次出现的日期，作为视频的发布日期
feed_release_date = user_action_df.groupby(['feedid'])['date_'].apply(lambda x: x.min())
feed_release_date = pd.DataFrame(feed_release_date)
feed_info_modified_df = pd.merge(feed_info_modified_df, feed_release_date, how='left', on='feedid')
# 将没有在之前出现过的视频发布日期设置为15
feed_info_modified_df['date_'].fillna(15, inplace=True)
feed_info_modified_df.rename(columns={'date_': 'release_date_'}, inplace=True)
feed_info_df = feed_info_modified_df.copy()
feed_info_df.set_index(['feedid'], inplace=True)
# 统一play和stay的单位：
user_action_df['play'] = user_action_df['play'] / 1000.0
user_action_df['stay'] = user_action_df['stay'] / 1000.0
# 标记用户是否产生过反馈：

user_action_df['feedback'] = (user_action_df[LABEL_COLUMNS].sum(axis=1) > 0).astype(np.int)
user_action_df['interested'] = (user_action_df[['favorite', 'read_comment', 'comment', 'like']].sum(axis=1) > 0).astype(
    np.int)
unique_user_df = pd.DataFrame(user_action_df['userid'].drop_duplicates())

# ===========================================================================================================================================
# 用户兴趣挖掘中三种 id 类特征挖掘可以采用完全相同的方式：
def User_Persona_ID(user_action_df, id, action_list):
    user_id_info = user_action_df.groupby(['userid', id]).size().reset_index()
    user_watch_sum = user_action_df.groupby('userid').size().reset_index()
    user_id_info = pd.merge(user_id_info, user_watch_sum, on='userid', how='left')
    user_id_info.rename(columns={'0_x': id + '_watched', '0_y': 'watched_sum'}, inplace=True)
    user_id_info[id + '_watched_rate'] = user_id_info[id + '_watched'] / user_id_info['watched_sum']
    for action in action_list:
        user_id_action = user_action_df[user_action_df[action] == 1].groupby(['userid', id]).size().reset_index()
        user_action_sum = user_action_df[user_action_df[action] == 1].groupby('userid').size().reset_index()
        user_id_action = pd.merge(user_id_action, user_action_sum, on='userid')
        user_id_action.rename(columns={'0_x': id + '_' + action, '0_y': id + '_' + action + '_sum'}, inplace=True)
        user_id_action[id + '_' + action + '_rate'] = user_id_action[id + '_' + action] / user_id_action[
            id + '_' + action + '_sum']
        user_id_info = pd.merge(user_id_info, user_id_action, on=['userid', id], how='left')
        user_id_info[id + '_' + action + '_partition'] = user_id_info[id + '_' + action] / user_id_info[id + '_watched']
    return user_id_info


# 对用户历史观看视频的长度进行挖掘，分别挖掘出：用户观看过、用户有过action反馈的视频时间长短的均值和中位数
def User_Persona_Videoplayseconds(user_action_df, action_list):
    user_videoplayseconds_info = user_action_df.groupby('userid')['videoplayseconds'].mean().reset_index()
    user_videoplayseconds_info.rename(columns={'videoplayseconds': 'watch_videoplayseconds_mean'}, inplace=True)
    user_videoplayseconds_info = pd.merge(user_videoplayseconds_info,
                                          user_action_df.groupby('userid')['videoplayseconds'].median().reset_index(),
                                          on='userid',
                                          how='left')
    user_videoplayseconds_info.rename(columns={'videoplayseconds': 'watch_videoplayseconds_median'}, inplace=True)
    for action in action_list:
        user_videoplayseconds_info = pd.merge(user_videoplayseconds_info,
                                              user_action_df[user_action_df[action] == 1].groupby('userid')[
                                                  'videoplayseconds'].mean().reset_index(),
                                              on='userid',
                                              how='left')
        user_videoplayseconds_info.rename(columns={'videoplayseconds': action + '_videoplayseconds_mean'}, inplace=True)
        user_videoplayseconds_info = pd.merge(user_videoplayseconds_info,
                                              user_action_df[user_action_df[action] == 1].groupby('userid')[
                                                  'videoplayseconds'].median().reset_index(),
                                              on='userid',
                                              how='left')
        user_videoplayseconds_info.rename(columns={'videoplayseconds': action + '_videoplayseconds_median'},
                                          inplace=True)
    return user_videoplayseconds_info


from gensim.models import TfidfModel
from gensim.corpora import Dictionary


def create_feed_keyword_profile(user_hist_info):
    dataset = user_hist_info['keyword_list'].values
    from gensim.corpora import Dictionary
    dct = Dictionary(dataset)
    corpus = [dct.doc2bow(line) for line in dataset]
    model = TfidfModel(corpus)
    _feed_keywords = []
    _feed_keywords_weights = []
    for i in range(len(corpus)):
        vector = model[corpus[i]]
        feed_keywords = sorted(vector, key=lambda x: x[1], reverse=True)
        keywords_weights = dict(map(lambda x: (dct[x[0]], x[1]), feed_keywords))
        keywords = [i[0] for i in keywords_weights.items()]
        _feed_keywords.append(keywords)
        _feed_keywords_weights.append(keywords_weights)
    return _feed_keywords, _feed_keywords_weights


# 分别计算feed的各个tag_list的权重
def create_feed_tags_profile(user_hist_info):
    dataset = user_hist_info['tag_list'].values
    from gensim.corpora import Dictionary
    dct = Dictionary(dataset)
    corpus = [dct.doc2bow(line) for line in dataset]
    model = TfidfModel(corpus)
    _tags = []
    _tags_weights = []
    for i in range(len(corpus)):
        vector = model[corpus[i]]
        feed_tags = sorted(vector, key=lambda x: x[1], reverse=True)
        tags_weights = dict(map(lambda x: (dct[x[0]], x[1]), feed_tags))
        tags = [i[0] for i in tags_weights.items()]
        _tags.append(tags)
        _tags_weights.append(tags_weights)
    return _tags, _tags_weights


def User_Hist_Interest(split_user_action_df):
    user_interested_df = split_user_action_df[split_user_action_df['feedback'] > 0].groupby('userid')[
        ['keyword_list', 'tag_list']].sum().reset_index()
    _feed_keywords, _feed_keywords_weights = create_feed_keyword_profile(user_interested_df)
    user_interested_df['hist_keywords'] = _feed_keywords
    user_interested_df['hist_keywords_weights'] = _feed_keywords_weights
    tags, tags_weights = create_feed_tags_profile(user_interested_df)
    user_interested_df['hist_tag'] = tags
    user_interested_df['hist_tag_weights'] = tags_weights
    user_interested_df.drop(columns=['keyword_list', 'tag_list'], inplace=True)
    print(type(user_interest_df))
    user_interested_df = pd.merge(unique_user_df, user_interested_df, on='userid', how='left')
    user_interested_df['hist_tag'] = user_interested_df['hist_tag'].apply(lambda x: x if type(x) != float else [])
    user_interested_df['hist_keywords'] = user_interested_df['hist_keywords'].apply(
        lambda x: x if type(x) != float else [])
    user_interested_df['hist_keywords_weights'] = user_interested_df['hist_keywords_weights'].apply(
        lambda x: x if type(x) != float else {})
    user_interested_df['hist_tag_weights'] = user_interested_df['hist_tag_weights'].apply(
        lambda x: x if type(x) != float else {})
    return user_interested_df


# 挖掘用户历史的 embedding 信息
def User_Persona_Embedding(user_action_df, action_list):
    def feed_embedding_mean(feedid):
        return feed_info_df.loc[feedid]['feed_embedding'].mean()

    user_embedding_df = user_action_df.groupby('userid')['feedid'].unique().reset_index()
    user_embedding_df['feed_embedding'] = user_embedding_df['feedid'].apply(lambda x: feed_embedding_mean(x))
    user_embedding_df.rename(columns={'feed_embedding': 'watch_embedding'}, inplace=True)
    for action in action_list:
        user_action_embedding_df = user_action_df[user_action_df[action] == 1].groupby('userid')[
            'feedid'].unique().reset_index()
        user_action_embedding_df['feed_embedding'] = user_action_embedding_df['feedid'].apply(
            lambda x: feed_embedding_mean(x))
        user_action_embedding_df.drop(columns=['feedid'], inplace=True)
        user_embedding_df = pd.merge(user_embedding_df, user_action_embedding_df, on='userid', how='left')
        user_embedding_df.rename(columns={'feed_embedding': action + '_embedding'}, inplace=True)
    return user_embedding_df


# 挖掘每个视频相关的id类特征的信息
def Feed_Persona_Id(user_action_df, id, action_list):
    id_info = user_action_df.groupby(id).size().reset_index()
    id_info.rename(columns={0: 'feed_' + id + '_occurs_times'}, inplace=True)
    for action in action_list:
        id_action_info = user_action_df[user_action_df[action] == 1].groupby(id).size().reset_index()
        id_action_info.rename(columns={0: 'feed_' + id + '_' + action + '_times'}, inplace=True)
        # print(id_action_info.head())
        id_info = pd.merge(id_info, id_action_info, on=id, how='left')
        id_info['feed_' + id + '_' + action + '_rate'] = id_info['feed_' + id + '_' + action + '_times'] / id_info[
            'feed_' + id + '_occurs_times']
    return id_info


# 用户行为习惯的挖掘
def User_Habit(user_action_df, action_list):
    user_habit_info = user_action_df.groupby('userid').size().reset_index()
    user_habit_info.rename(columns={0: 'past_watch_sum'}, inplace=True)
    for action in action_list:
        user_action = user_action_df[user_action_df[action] == 1].groupby('userid').size().reset_index()
        user_action.rename(columns={0: action + '_sum'}, inplace=True)
        user_habit_info = pd.merge(user_habit_info, user_action, on='userid', how='left')
        user_habit_info[action + '_rate'] = user_habit_info[action + '_sum'] / user_habit_info['past_watch_sum']
    return user_habit_info


# device信息的挖掘
def Device_Info(user_action_df, action_list):
    device_info = user_action_df.groupby('device').size().reset_index()
    device_info.rename(columns={0: 'device_sum'}, inplace=True)
    for action in action_list:
        device_action = user_action_df[user_action_df[action] == 1].groupby('device').size().reset_index()
        device_action.rename(columns={0: 'device_' + action + '_sum'}, inplace=True)
        device_info = pd.merge(device_info, device_action, on='device', how='left')
        device_info['device_' + action + '_rate'] = device_info['device_' + action + '_sum'] / device_info['device_sum']
    return device_info


# 处理用户历史行为数据：
def ProcessingUserAction(user_action_test, feed_info_modified_df):
    user_action_test = pd.merge(user_action_test, feed_info_modified_df, on='feedid')
    user_action_test['is_finished'] = (user_action_test['play'] >= user_action_test['videoplayseconds']).astype(np.int)
    user_action_test['is_stay'] = (user_action_test['stay'] >= user_action_test['videoplayseconds'] * 2.0).astype(
        np.int)
    user_author_info = User_Persona_ID(user_action_test, 'authorid', ACTION_LIST)
    user_feed_info = user_action_test.groupby(['userid', 'feedid']).size().reset_index()
    user_feed_info.rename(columns={0: 'user_feedid_occurs'}, inplace=True)
    user_videoplayseconds_info = User_Persona_Videoplayseconds(user_action_test, ACTION_LIST)
    user_embedding_info = User_Persona_Embedding(user_action_test, ACTION_LIST)
    user_hist_interest_info = User_Hist_Interest(user_action_test)
    user_embedding_info.drop(columns=['feedid'], inplace=True)
    feedid_info = Feed_Persona_Id(user_action_test, 'feedid', ACTION_LIST)
    feed_author_info = Feed_Persona_Id(user_action_test, 'authorid', ACTION_LIST)
    user_habit_info = User_Habit(user_action_test, ACTION_LIST)
    device_info = Device_Info(user_action_test, ACTION_LIST)
    return user_feed_info, user_author_info, user_videoplayseconds_info, user_embedding_info, user_hist_interest_info, feedid_info, feed_author_info, user_habit_info, device_info


def keyword_tag_coocurrence(past_hist_list):
    coocurrence = []
    for past, hist in past_hist_list:
        cor = 0
        for word in past:
            if word in hist:
                cor += 1
        coocurrence.append(cor)
    return coocurrence


def keyword_tag_weights(past_list_hist_weight):
    coocurrence_weights = []
    for past, hist_weights in past_list_hist_weight:
        weight = 0
        for word in past:
            weight += hist_weights.get(word, 0)
        coocurrence_weights.append(weight)
    return coocurrence_weights


# 挖掘数据历史行为信息并将其作为下一天的特征
def DataProcessing(user_hist_actions, user_tag_test, feed_info_modified, date):
    user_feed_info, user_author_info, user_videoplayseconds_info, user_embedding_info, user_hist_interest_info, feedid_info, feed_author_info, user_habit_info, device_info = ProcessingUserAction(
        user_hist_actions, feed_info_modified)
    user_tag_test['date_'] = date
    user_tag_test = pd.merge(user_tag_test, feed_info_modified, on='feedid', how='left')
    # 将 user_feed_info 拼接
    user_tag_test = pd.merge(user_tag_test, user_feed_info, on=['userid', 'feedid'], how='left')
    user_tag_test['user_feedid_occurs'] = user_tag_test['user_feedid_occurs'].fillna(0)
    # 将 user_author_info 拼接，并将缺失值置为0
    user_tag_test = pd.merge(user_tag_test, user_author_info, on=['userid', 'authorid'], how='left')
    user_tag_test.drop(columns=['watched_sum'], inplace=True)
    columns = list(user_author_info.columns.values)
    for column in ['userid', 'authorid', 'watched_sum']:
        columns.remove(column)
    user_tag_test[columns] = user_tag_test[columns].fillna(0)
    # 将 user_videoplayseconds_info 拼接，并转化为计算和视频长度的差距
    user_tag_test = pd.merge(user_tag_test, user_videoplayseconds_info, on='userid', how='left')
    columns = list(user_videoplayseconds_info.columns.values)
    columns.remove('userid')
    '''
    for column in columns:
        user_tag_test[column] = 1.0 / np.exp(
            abs(user_tag_test[column] - user_tag_test['videoplayseconds']) / (0.0001+user_tag_test['videoplayseconds']))
    '''
    # 计算用户历史关键词和视频关键词之间的关系
    user_tag_test = pd.merge(user_tag_test, user_hist_interest_info, on='userid', how='left')
    user_tag_test['tag_cooccurrence'] = keyword_tag_coocurrence(user_tag_test[['tag_list', 'hist_tag']].values)
    user_tag_test['tag_cooccurrence_weights'] = keyword_tag_weights(
        user_tag_test[['tag_list', 'hist_tag_weights']].values)
    user_tag_test['keyword_cooccurrence'] = keyword_tag_coocurrence(
        user_tag_test[['keyword_list', 'hist_keywords']].values)
    user_tag_test['keyword_cooccurrence_weights'] = keyword_tag_weights(
        user_tag_test[['keyword_list', 'hist_keywords_weights']].values)
    # 视频特征和其他特征集中处理：
    other_id_info = [feedid_info, feed_author_info, user_habit_info, device_info]
    columns_to_drop = ['feedid', 'authorid', 'userid', 'device']
    for index, (id_info, column) in enumerate(zip(other_id_info, columns_to_drop)):
        user_tag_test = pd.merge(user_tag_test, id_info, on=column, how='left')
        columns = list(id_info.columns.values)
        columns.remove(column)
        user_tag_test[columns] = user_tag_test[columns].fillna(0)

    # 处理 embedding 特征，计算相似度
    def embedding_similarity(x):
        """
        计算余弦相似度距离
        """
        result = []
        for pair in x:
            result.append(distance.cosine(pair[0], pair[1]))
        return result

    def embedding_distance(x):
        """
        计算欧氏距离
        """
        result = []
        for pair in x:
            result.append(np.sqrt(np.sum((pair[0] - pair[1]) ** 2)))
        return result

    user_tag_test = pd.merge(user_tag_test, user_embedding_info, on='userid', how='left')
    columns = list(user_embedding_info.columns.values)
    columns.remove('userid')
    for column in columns:
        user_tag_test[column + '_dist'] = embedding_distance(user_tag_test[[column, 'feed_embedding']].values)
        user_tag_test[column + '_dist'] = user_tag_test[column + '_dist'].fillna(0)
        user_tag_test[column + '_sim'] = embedding_similarity(user_tag_test[[column, 'feed_embedding']].values)
        user_tag_test[column + '_sim'] = user_tag_test[column + '_sim'].fillna(0)
    # 视频热度信息（距离首次出现的时间）
    user_tag_test['date_gap'] = user_tag_test['date_'] - user_tag_test['release_date_']
    # 丢弃多余的行
    drop_columns1 = list(user_embedding_info.columns)
    drop_columns1.remove('userid')
    drop_columns2 = list(user_hist_interest_info.columns)
    drop_columns2.remove('userid')
    drop_columns = ['keyword_list', 'tag_list', 'feed_embedding', 'date_']
    drop_columns += drop_columns1
    drop_columns += drop_columns2
    user_tag_test.drop(columns=drop_columns, inplace=True)
    del user_feed_info, user_author_info, user_videoplayseconds_info, user_embedding_info, user_hist_interest_info, feedid_info, feed_author_info, user_habit_info, device_info
    return user_tag_test


def Split_and_Process(user_action_df, days):
    """
    days: 滑动窗口的大小
    """
    print('start processing')
    for date in range(14 - days):
        user_action_train_split = user_action_df[
            (user_action_df['date_'] >= 1) & (user_action_df['date_'] <= date + days)]
        user_action_val_split = user_action_df[user_action_df['date_'] == date + days + 1][
            ['userid', 'feedid', 'device'] + LABEL_COLUMNS]
        user_action_train_processed = DataProcessing(user_action_train_split, user_action_val_split,
                                                     feed_info_modified_df, date + days + 1)
        user_action_train_processed.to_csv(
            os.path.join(RootPath, "user_action_" + str(date + days + 1) + '_version2.csv'))
        del user_action_train_split, user_action_val_split, user_action_train_processed
        print('{} finished'.format(date))
    user_action_train_split = user_action_df[(user_action_df['date_'] >= 1) & (user_action_df['date_'] <= 14)].copy()
    user_action_val_split = test_data_df
    user_action_train_processed = DataProcessing(user_action_train_split, user_action_val_split, feed_info_modified_df,
                                                 15)
    user_action_train_processed.to_csv(os.path.join(RootPath, "test_version2.csv"))

if __name__=="__main__":
    pd.set_option('display.max_columns', None)
    #user_embedding_df = user_action_df.groupby('userid')['feedid'].unique().reset_index()
    #print(user_embedding_df.head())
    #print(user_action_df.columns)
    #print(user_action_df.columns)
    #print(user_interest_df.columns)
    #print(feed_info_modified_df.columns)
    #Split_and_Process(user_action_df, 7)
    #user_test_action_df = user_action_df[(user_action_df['date_'] >= 7) & (user_action_df['date_'] <= 13)].copy()
    #user_tag_test_df = user_action_df[user_action_df['date_'] == 14][['userid', 'feedid', 'device', 'date_']].copy()
    #processed_data = DataProcessing(user_test_action_df, user_tag_test_df, feed_info_modified_df, 14)
    #print(feed_info_modified_df.columns)
    print(feed_info_df.iloc[0:2]['feed_embedding'].mean())

