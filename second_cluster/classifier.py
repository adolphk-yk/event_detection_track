import pickle
import random
import jieba
from first_cluster.first_cluster import load_stop_words
import gensim
from gensim import corpora,models,similarities
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.externals import joblib


def load_data(pos_data_path='../temp/second_cluster/positive.txt', neg_data_path='../temp/second_cluster/negative.txt'):
    # read positive and negative sample file
    pos_data_all = []
    neg_data_all = []
    news_path = '../temp/news.pkl'
    with open(pos_data_path, 'r', encoding='utf-8') as fin:
        for single in fin:
            pos_data_all.append(single.strip('\n').split(' '))
    with open(neg_data_path, 'r', encoding='utf-8') as fin:
        for single in fin:
            neg_data_all.append(single.strip('\n').split(' '))
    # load news infomation
    with open(news_path, 'rb') as fout:
        news_dict = pickle.load(fout)
    # sample data
    pos_data = random.sample(pos_data_all, 8000)
    neg_data = random.sample(neg_data_all, 20000)

    pos_neg_data = []
    pos_neg_data.extend(pos_data)
    pos_neg_data.extend(neg_data)
    news_id_all = []
    for single in pos_neg_data:
        news_id_all.append(single[0])
        news_id_all.append(single[1])
    news_id_all = set(news_id_all)
    return news_id_all, news_dict, pos_data, neg_data


def cut_with_stopword(text, stop_words):
    temp_cut = []
    for single_cut in jieba.lcut(text):
        if (single_cut not in stop_words) and len(single_cut)>1:
            temp_cut.append(single_cut)
    return temp_cut

def com_simi(text_1, text_2, num_feature):
    index = similarities.SparseMatrixSimilarity([text_1], num_features=num_feature)
    result = index[text_2]
    return result[0]

def get_feature(dct, model, news_related_dict, data, feature_list=['newsTitle', 'digest', 'content']):
    feature = []
    num_features = len(dct.token2id.keys())
    for single in tqdm(data):
        temp_feature = []
        news_1 = news_related_dict[single[0]]
        news_2 = news_related_dict[single[1]]
        for single_feature in feature_list:
            news_1_tf = dct.doc2bow(news_1[single_feature])
            news_2_tf = dct.doc2bow(news_2[single_feature])
            temp_feature.append(com_simi(news_1_tf, news_2_tf, num_features))
            news_1_tfidf = model[news_1_tf]
            news_2_tfidf = model[news_2_tf]
            temp_feature.append(com_simi(news_1_tfidf, news_2_tfidf, num_features))

        feature.append(temp_feature)
    return feature


def build(news_id_all, news_dict, pos_data, neg_data):
    print('===build data feature===')
    stop_words = load_stop_words()
    news_related_dict = {}
    corpus = []
    for single_news_id in news_id_all:
        news_related_dict[single_news_id] = {}
        newsTitle = news_dict[single_news_id]['newsTitle']
        digest = news_dict[single_news_id]['digest']
        content = news_dict[single_news_id]['content'].split('。')[0]
        news_related_dict[single_news_id]['newsTitle'] = cut_with_stopword(newsTitle, stop_words)
        news_related_dict[single_news_id]['digest'] = cut_with_stopword(digest, stop_words)
        news_related_dict[single_news_id]['content'] = cut_with_stopword(content, stop_words)
        news_related_dict[single_news_id]['all'] = news_related_dict[single_news_id]['newsTitle'] + \
                                                   news_related_dict[single_news_id]['digest'] + \
                                                   news_related_dict[single_news_id]['content']
        corpus.append(news_related_dict[single_news_id]['all'])

    dct = Dictionary(corpus)
    corpus2id = [dct.doc2bow(line) for line in corpus]
    tf_idf_model = TfidfModel(corpus2id)

    pos_data_feature = get_feature(dct, tf_idf_model, news_related_dict, pos_data)
    neg_data_feature = get_feature(dct, tf_idf_model, news_related_dict, neg_data)

    useful_pos_data_feature = []
    useful_neg_data_feature = []
    for single in pos_data_feature:
        if sum(single) != 0:
            useful_pos_data_feature.append(single)
    for single in neg_data_feature:
        if sum(single) != 0:
            useful_neg_data_feature.append(single)
    print('positive sample data %d' % len(useful_pos_data_feature))
    print('negative sample data %d' % len(useful_neg_data_feature))

    return useful_pos_data_feature, useful_neg_data_feature


def train_classifier(pos_data, neg_data, c=2, kernel='rbf', gamma=5, model_save_path='../temp/second_cluster/svm_clf.model'):
    print('===train classifier')
    train_data = pos_data + neg_data
    train_target = [1] * len(pos_data) + [0] * len(neg_data)
    x_train, x_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.2, random_state=666)
    clf = SVC(C=c, kernel=kernel, gamma=gamma, probability=True)
    clf.fit(x_train, y_train)
    train_score = clf.score(x_train, y_train)
    print('train acc: %f' % train_score)
    test_score = clf.score(x_test, y_test)
    print('test acc: %f' % test_score)
    joblib.dump(clf, model_save_path)
    print('Model has been saved at %s' % model_save_path)


def run_classifier():
    news_id_all, news_dict, pos_data, neg_data = load_data()
    useful_pos_data_feature, useful_neg_data_feature = build(news_id_all, news_dict, pos_data, neg_data)
    train_classifier(useful_pos_data_feature, useful_neg_data_feature)


if __name__ == '__main__':
    run_classifier()
