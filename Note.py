'''
这篇 notebook 的主要作用是根据用户的历史行为信息进行用户画像,主要是挖掘用户的兴趣信息，然后分开为训练集和测试集输入到树模型中进行；
具体方式为:
  根据前7天的历史行为数据，对用户和物品进行进一步地描述和挖掘（也可以之后尝试不是7天的历史行为数据，而是根据所有历史行为数据进行挖掘）
  然后将挖掘到的信息，作为第8天的特征，将第八天的数据输入到树模型中进行训练；
  数据集一共包含了14天的数据，于是可以分别划分为8份，分别是：
  1. 挖掘 1 到 7 天的信息，作为第 8 天数据的特征；
  2. 挖掘 2 到 8 天的信息，作为第 9 天数据的特征；
  3. 挖掘 3 到 9 天的信息，作为第 10 天数据的特征；
  4. 挖掘 4 到 10 天的信息，作为第 11 天数据的特征；
  5. 挖掘 5 到 11 天的信息，作为第 12 天数据的特征；
  6. 挖掘 6 到 12 天的信息，作为第 13 天数据的特征；
  7. 挖掘 7 到 13 天的信息，作为第 14 天数据的特征；
  8. 挖掘 8 到 14 天的信息，作为第 15 天数据的特征；
  在线下训练阶段，选在第 8 天到第 13 天的数据作为训练集，而第 14 天的数据做验证；
  当需要test数据进行预测时候，则直接输入第 8 天到第 14 天的数据进行训练，而test是第 15 天的数据；

需要挖掘的历史行为数据包括以下几个部分：
1.用户兴趣挖掘，根据用户过去7天的行为数据，挖掘用户对于feed的偏好信息，然后将这些偏好信息或者直接输入模型，或者和对应的feed特征进行一些交叉；
  当前视频的特征包括：authorid、videoplaysecond、bgm_song_id、bgm_singer_id、author_n_feeds、top5_keywords、top5_keywords_weights、tag_list、tag_weights、feed_embedding
  对于这些特征，可以依次挖掘出一些信息：
  authorid：挖掘此信息主要是挖掘出用户对于author的偏好，主要可以挖掘的包括：
  ·观看历史中，authorid出现的次数；
  ·观看历史中，authorid所占的比例；
  ·观看历史中，authorid出现的次数占author作者所有作品的比例
  ·用户所有完整播放甚至重复播放的视频中，authorid出现的次数；
  ·用户所有完整播放甚至重复播放的视频中，authorid所占的比例；
  ·用户所有完整播放甚至重复播放的视频中，authorid占authorid出现次数的比例；
  ·用户停留时间很长的视频中，authorid出现的次数；
  ·用户停留时间很长的视频中，authorid所占的比例；
  ·用户停留时间很长的视频中，authorid占authorid出现次数的比例
  ·某一具体的历史action中，authorid出现的次数；
  ·某一具体的历史action中，authorid所占的比例；
  ·某一具体的历史action中，authorid占authorid出现次数的比例
  ·对于和某一action强相关的特征，也按照上面3个进行统计；
    - click_avatar 统计 follow
    - like 统计 follow、comment
    - read_comment 统计 favorite、comment
  以上挖掘出的authorid特征都需要和视频的authorid做交叉，也就是只需要视频authorid对应的值就行了；
  bgm_song_id：挖掘此信息是为了挖掘出用户对于 bgm_song 的偏好，挖掘过程和authorid的挖掘过程相似；
  bgm_singer_id：挖掘此信息是为了挖掘出用户对于 bgm_singer 的偏好，挖掘过程和authorid的挖掘过程相似；
  videoplayseconds：视频时长信息，挖掘此信息是为了挖掘出用户对于视频市场的偏好信息；
  ·用户观看历史中，视频时间长度的均值、中位数、分位数均值（前25% 到 50%的均值）
  ·用户历史action中，视频时长的均值、中位数、分位数均值（前25% 到 50%的均值）
  ·和某一action强相关的action中，视频时长的均值、中位数、分位数均值（前25% 到 50%的均值）。
  ·用户所有完整播放的视频中，视频时长的均值、中位数、分位数均值（前25% 到 50%的均值）
  ·用户所有停留时间很长的视频中，视频时长的均值、中位数、分位数均值（前25% 到 50%的均值）
  上述特征的值均可以和视频时长做交叉，具体思路是，上面得到的各个值和视频时长之间的差距，为了衡量，可以采用 1/exp（|得到的值-视频时长|/视频时长）) 表示相关度、之后可以分别将多个相关度相乘做交叉；最终衡量时间上的用户偏好；
  top5_keywords和top5_keywords_weights: 挖掘此信息是为了挖掘用户对于关键词的偏好程度，也就是可以挖掘用户对于什么样关键词的视频更喜欢；
  ·用户历史观看视频中，keywords出现的比例和的权重，由各个视频的keywords相加的到；
  ·用户历史action视频中，keywords出现的比例和权重，有各个视频的keywords相加得到；
  ·和某一action强相关的action中，keywords出现的比例和权重，由各个视频的keywords相加得到；
  ·用户历史完整播放的action中，keywords出现的比例和权重，由各个视频的keywords相加得到；
  ·用户历史停留时间长的视频中，keywords出现的比例和权重，由各个视频的keywords统计得到；
  对于上述得到的用户兴趣，需要和视频做交叉，也就是只取视频的keywords对应出现的比例相加以及将视频keywords权重和用户兴趣权重相乘
  再可以将所有的相关权重或者出现次数融合成一个关键词的总特征；
  tag_list和tag_weights：挖掘此信息是为了挖掘用户对于视频标签的偏好，也就是挖掘用户喜欢什么样的视频；挖掘方式和前面关键词的挖掘类似即可；最终也需要将其和具体视频的特征做交叉；
  feed_embedding挖掘：挖掘此信息是为了挖掘用户对于和给出视频相似的视频的偏好信息；
  ·用户历史观看的视频embedding均值
  ·用户历史action的视频embedding均值；
  ·用户历史和某一action强相关的action的视频embedding均值
  ·用户完整播放的视频embedding均值
  ·用户停留时间长的embedding均值
  上面得到的embedding都需要再和对应视频的embedding算欧氏距离或者是余弦相似度；
  所有的欧氏距离或者余弦相似度可以再一次计算出feed_embedding相似度的总信息；

2.过去7天内，视频的特征，这是和具体的用户无关的，而只是视频的特征信息，最后或者作为特征直接输入，或者和用户兴趣做一些交叉；
  feedid：挖掘这个特征主要是挖掘某个视频在过去7天之内流行度信息等等；
  ·过去7天之内，该视频出现的次数；
  ·过去7天之内，该视频某一action的次数以及占其出现次数的比例；
  ·过去7天之内，该视频在和某一action相关性最大的action次数以及占其出现次数的比例；
  ·过去7天之内，该视频被完整播放的次数以及占其出现次数的比例；
  authorid：挖掘这个特征的主要目的是挖掘视频作者的受欢迎程度；
  ·过去7天之内，该视频作者的作品出现的次数；
  ·过去7天之内，该视频作者的作品被action的次数，以及被action所占的比例；
  ·过去7天之内，该视频作者的作品被完整播放的次数，以及被完整播放的比例；
  bgm_song_id：挖掘该特征的目的是挖掘视频bgm的受欢迎程度等等，和authorid的处理比较类似；
  ·过去7天之内，该bgm_song_id出现的次数；
  ·过去7天之内，该bgm_song_id的作品被action的次数，以及被action所占的比例；
  ·过去7天之内，该bgm_song_id的视频被完整播放的次数，以及完整播放所占的比例；
  bgm_singer_id：挖掘该特征的目的是挖掘 bgm_singer 的受欢迎程度；
  ·过去7天之内，该bgm_singer_id出现的次数；
  ·过去7天之内，该bgm_singer_id的作品被action的次数，以及被action所占的比例；
  ·过去7天之内，该bgm_singer_id的视频被完整播放的次数以及完整播放所占的比例；

3.其他信息的一些挖掘；
  用户的历史行为习惯，主要是用户的历史行为action占用户所有用户观看历史的比例、以及用户完整播放视频所占的比例等；
  视频发布时间信息，默认视频第一次出现的时间是其发布时间，可以加上视频发布时间距离最近的时间；
  device信息的挖掘，包括不同的action中，各种device所占的比例，或者不同device的action所占action的总数比例；以及用户不同的action中，device所占的比例等；

4.在之前的特征基础上所做的交叉等等；
特征表：
userid
feedid
device
click_avatar
forward
follow
favorite
read_comment
comment
like
authorid
videoplayseconds
bgm_song_id
bgm_singer_id
以上为源数据的特征
author_n_feeds   作者的作品数 feedinfo 里对authorid 进行value_count
videolength_bucket 将视频以10s为单位分桶 视频都在80s以内
embedding_0
embedding_1
embedding_2
embedding_3
embedding_4
embedding_5
embedding_6
embedding_7
embedding_8
embedding_9
embedding_10
embedding_11
embedding_12
embedding_13
embedding_14
embedding_15
embedding_16
embedding_17
embedding_18
embedding_19
embedding_20
embedding_21
embedding_22
embedding_23
embedding_24
embedding_25
embedding_26
embedding_27
embedding_28
embedding_29
embedding_30
embedding_31  embedding是通过pca降维得到的
author_n_feeds_bucket 对作者作品数取log然后取整分桶 不合理
release_date_ 视频首次出现的时间作为发布时间
user_feedid_occurs 最近7天该feed对该用户的曝光次数
authorid_watched 用户曝光过该author的多少视频
authorid_watched_rate 对某一用户该author的曝光视频所占比例
authorid_click_avatar 在该作者前提下 该用户点击该头像的次数
authorid_click_avatar_sum 该用户点击 头像总次数
authorid_click_avatar_rate 用户点击该作者头像次数占总点击次数的比例
authorid_click_avatar_partition 该作者 对应的点击次数/曝光次数
authorid_forward
authorid_forward_sum
authorid_forward_rate
authorid_forward_partition
authorid_follow
authorid_follow_sum
authorid_follow_rate
authorid_follow_partition
authorid_favorite
authorid_favorite_sum
authorid_favorite_rate
authorid_favorite_partition
authorid_read_comment
authorid_read_comment_sum
authorid_read_comment_rate
authorid_read_comment_partition
authorid_comment
authorid_comment_sum
authorid_comment_rate
authorid_comment_partition
authorid_like
authorid_like_sum
authorid_like_rate
authorid_like_partition
authorid_is_stay
authorid_is_stay_sum
authorid_is_stay_rate
authorid_is_stay_partition
authorid_is_finished
authorid_is_finished_sum
authorid_is_finished_rate
authorid_is_finished_partition
authorid_feedback
authorid_feedback_sum
authorid_feedback_rate
authorid_feedback_partition
authorid_interested
authorid_interested_sum
authorid_interested_rate
authorid_interested_partition
注意到LABEL_COLUMNS = ['click_avatar', 'forward', 'follow', 'favorite', 'read_comment', 'comment', 'like']
feed_back表示有以上任意行为的
user_action_df['is_finished'] = (user_action_df['play'] >= user_action_df['videoplayseconds'] * 0.9).astype(np.int)
user_action_df['is_stay'] = (user_action_df['stay'] - user_action_df['videoplayseconds'] >= 10).astype(np.int)
user_action_df['interested'] = (user_action_df[['favorite', 'read_comment', 'comment', 'like']].sum(axis=1) > 0).astype(np.int)
以上内容全部同上
watch_videoplayseconds_mean
watch_videoplayseconds_median
click_avatar_videoplayseconds_mean
click_avatar_videoplayseconds_median
forward_videoplayseconds_mean
forward_videoplayseconds_median
follow_videoplayseconds_mean
follow_videoplayseconds_median
favorite_videoplayseconds_mean
favorite_videoplayseconds_median
read_comment_videoplayseconds_mean
read_comment_videoplayseconds_median
comment_videoplayseconds_mean
comment_videoplayseconds_median
like_videoplayseconds_mean
like_videoplayseconds_median
is_stay_videoplayseconds_mean
is_stay_videoplayseconds_median
is_finished_videoplayseconds_mean
is_finished_videoplayseconds_median
feedback_videoplayseconds_mean
feedback_videoplayseconds_median
interested_videoplayseconds_mean
interested_videoplayseconds_median
以上特征意义不大
视频的keyword 和tag 都是用TfidfModel算出来的相似度
tag_cooccurrence 统计视频tag在用户tag里的数量
tag_cooccurrence_weights 用户的tag是有权重的 求视频tag对应的权重之和 weight += hist_weights.get(word, 0)
keyword_cooccurrence
keyword_cooccurrence_weights
feed_feedid_occurs_times 视频的曝光总次数
feed_feedid_click_avatar_times 点击次数
feed_feedid_click_avatar_rate 点击率
feed_feedid_forward_times
feed_feedid_forward_rate
feed_feedid_follow_times
feed_feedid_follow_rate
feed_feedid_favorite_times
feed_feedid_favorite_rate
feed_feedid_read_comment_times
feed_feedid_read_comment_rate
feed_feedid_comment_times
feed_feedid_comment_rate
feed_feedid_like_times
feed_feedid_like_rate
feed_feedid_is_stay_times
feed_feedid_is_stay_rate
feed_feedid_is_finished_times
feed_feedid_is_finished_rate
feed_feedid_feedback_times
feed_feedid_feedback_rate
feed_feedid_interested_times
feed_feedid_interested_rate
feed_authorid_occurs_times author的曝光次数
feed_authorid_click_avatar_times author的点击次数
feed_authorid_click_avatar_rate  author的点击率
feed_authorid_forward_times
feed_authorid_forward_rate
feed_authorid_follow_times
feed_authorid_follow_rate
feed_authorid_favorite_times
feed_authorid_favorite_rate
feed_authorid_read_comment_times
feed_authorid_read_comment_rate
feed_authorid_comment_times
feed_authorid_comment_rate
feed_authorid_like_times
feed_authorid_like_rate
feed_authorid_is_stay_times
feed_authorid_is_stay_rate
feed_authorid_is_finished_times
feed_authorid_is_finished_rate
feed_authorid_feedback_times
feed_authorid_feedback_rate
feed_authorid_interested_times
feed_authorid_interested_rate
以上特征有同质性
past_watch_sum 用户被曝光的视频数量
click_avatar_sum 用户的点击次数
click_avatar_rate 用户的点击率
forward_sum
forward_rate
follow_sum
follow_rate
favorite_sum
favorite_rate
read_comment_sum
read_comment_rate
comment_sum
comment_rate
like_sum
like_rate
is_stay_sum
is_stay_rate
is_finished_sum
is_finished_rate
feedback_sum
feedback_rate
interested_sum
interested_rate
device_sum device曝光的次数
device_click_avatar_sum device的点击总量
device_click_avatar_rate 点击率
device_forward_sum
device_forward_rate
device_follow_sum
device_follow_rate
device_favorite_sum
device_favorite_rate
device_read_comment_sum
device_read_comment_rate
device_comment_sum
device_comment_rate
device_like_sum
device_like_rate
device_is_stay_sum
device_is_stay_rate
device_is_finished_sum
device_is_finished_rate
device_feedback_sum
device_feedback_rate
device_interested_sum
device_interested_rate
其他特征
user侧的embedding是根据feed_embedding求均值计算出来的
分别计算了欧式距离和余弦距离
watch_embedding_dist
watch_embedding_sim
click_avatar_embedding_dist
click_avatar_embedding_sim
forward_embedding_dist
forward_embedding_sim
follow_embedding_dist
follow_embedding_sim
favorite_embedding_dist
favorite_embedding_sim
read_comment_embedding_dist
read_comment_embedding_sim
comment_embedding_dist
comment_embedding_sim
like_embedding_dist
like_embedding_sim
is_stay_embedding_dist
is_stay_embedding_sim
is_finished_embedding_dist
is_finished_embedding_sim
feedback_embedding_dist
feedback_embedding_sim
interested_embedding_dist
interested_embedding_sim
date_gap
目前的问题缺乏author的embedding
对keyword和tag的处理不够充分


user_interest的内容为Index(['userid', 'hist_tag', 'hist_keywords', 'hist_keywords_weights',
       'hist_tag_weights'],
      dtype='object')
平常使用feed_info_modified

Training until validation scores don't improve for 50 rounds.
[50]	valid_0's auc: 0.932926
[100]	valid_0's auc: 0.934089
[150]	valid_0's auc: 0.934908
[200]	valid_0's auc: 0.935499
[250]	valid_0's auc: 0.935889
[300]	valid_0's auc: 0.936119
[350]	valid_0's auc: 0.936278
[400]	valid_0's auc: 0.936395
[450]	valid_0's auc: 0.936485
[500]	valid_0's auc: 0.936547
[550]	valid_0's auc: 0.936581
[600]	valid_0's auc: 0.936649
[650]	valid_0's auc: 0.936678
[700]	valid_0's auc: 0.936736
[750]	valid_0's auc: 0.936742
[800]	valid_0's auc: 0.936751
[850]	valid_0's auc: 0.936782
[900]	valid_0's auc: 0.936791
[950]	valid_0's auc: 0.936795
[1000]	valid_0's auc: 0.936798
[1050]	valid_0's auc: 0.936819
[1100]	valid_0's auc: 0.936854
[1150]	valid_0's auc: 0.936868
[1200]	valid_0's auc: 0.936859
Early stopping, best iteration is:
[1153]	valid_0's auc: 0.936868
0.9368684268153984
read_comment 0.6438656456484296
Training until validation scores don't improve for 50 rounds.
[50]	valid_0's auc: 0.842858
[100]	valid_0's auc: 0.846604
[150]	valid_0's auc: 0.849221
[200]	valid_0's auc: 0.850978
[250]	valid_0's auc: 0.851958
[300]	valid_0's auc: 0.852603
[350]	valid_0's auc: 0.853093
[400]	valid_0's auc: 0.853397
[450]	valid_0's auc: 0.853564
[500]	valid_0's auc: 0.853661
[550]	valid_0's auc: 0.853752
[600]	valid_0's auc: 0.853822
[650]	valid_0's auc: 0.853954
[700]	valid_0's auc: 0.854006
[750]	valid_0's auc: 0.854047
[800]	valid_0's auc: 0.854101
[850]	valid_0's auc: 0.854135
[900]	valid_0's auc: 0.854157
Early stopping, best iteration is:
[880]	valid_0's auc: 0.854179
0.8541789167355744
like 0.6385393034989906
Training until validation scores don't improve for 50 rounds.
[50]	valid_0's auc: 0.855379
[100]	valid_0's auc: 0.859424
[150]	valid_0's auc: 0.861836
[200]	valid_0's auc: 0.863276
[250]	valid_0's auc: 0.864159
[300]	valid_0's auc: 0.864651
[350]	valid_0's auc: 0.865054
[400]	valid_0's auc: 0.865168
[450]	valid_0's auc: 0.865386
[500]	valid_0's auc: 0.865549
Early stopping, best iteration is:
[498]	valid_0's auc: 0.865556
0.8655555685476127
click_avatar 0.7355329814297339
Training until validation scores don't improve for 50 rounds.
[50]	valid_0's auc: 0.891685
[100]	valid_0's auc: 0.893647
[150]	valid_0's auc: 0.895051
[200]	valid_0's auc: 0.896378
[250]	valid_0's auc: 0.897008
[300]	valid_0's auc: 0.897422
[350]	valid_0's auc: 0.897577
[400]	valid_0's auc: 0.897826
[450]	valid_0's auc: 0.897669
Early stopping, best iteration is:
[404]	valid_0's auc: 0.897862
0.8978620488143462
forward 0.7217473833983137
0.6683893839348471
'''
if __name__=="__main__":
    import pandas as pd
    a={"userid":[1,2,3],"feedid":[2,3,4]}
    a=pd.DataFrame(a)
    print(a['userid'][0])
