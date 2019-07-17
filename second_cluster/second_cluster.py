import gensim
from gensim import corpora,models,similarities
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import jieba
from sklearn.externals import joblib
from itertools import combinations
from tqdm import tqdm
import networkx as nx
from networkx.algorithms import community
from collections import defaultdict
import pickle
from first_cluster.first_cluster import load_stop_words
from classifier import cut_with_stopword, com_simi


def load_data(used_news='../temp/first_cluster/used_event_news.pkl',
              first_cluster_result='../temp/first_cluster/result_community2news_id.pkl',
              news_path='../temp/news.pkl',
              classifier_path='../temp/second_cluster/svm_clf.model'):
    '''
    first cluster used data(ground truth)
    first cluster result(news id)
    news dict(include news content)
    :return:
    '''
    with open(used_news, 'rb') as fin:
        first_cluster_truth = pickle.load(fin)
    with open(first_cluster_result, 'rb') as fin:
        first_cluster_result = pickle.load(fin)
    with open(news_path, 'rb') as fin:
        news_dict = pickle.load(fin)
    # load svm clf model
    clf = joblib.load(classifier_path)

    return first_cluster_truth, first_cluster_result, news_dict, clf


def topic_corpus(topic_news_list, news_dict):
    used_news_content = {}
    corpus = []
    stop_words = load_stop_words()
    for single_news_id in topic_news_list:
        single_news_id_content_raw = news_dict[single_news_id]
        used_news_content[single_news_id] = {}
        used_news_content[single_news_id]['title'] = cut_with_stopword(single_news_id_content_raw['newsTitle'], stop_words)
        used_news_content[single_news_id]['digest'] = cut_with_stopword(single_news_id_content_raw['digest'], stop_words)
        first_sentence = single_news_id_content_raw['content'].split('。')[0]
        used_news_content[single_news_id]['content'] = cut_with_stopword(first_sentence, stop_words)
        used_news_content[single_news_id]['event_id'] = single_news_id_content_raw['specialId']
        corpus.append(used_news_content[single_news_id]['title']+used_news_content[single_news_id]['digest']+used_news_content[single_news_id]['content'])
    return used_news_content, corpus


def get_feature_vecture(news_id_1, news_id_2, dct, used_news_content, model, feature_list=['title', 'digest', 'content']):
    vec = []
    num_feature = len(dct.token2id.keys())
    for single_feature in feature_list:
        news_1_tf = dct.doc2bow(used_news_content[news_id_1][single_feature])
        news_2_tf = dct.doc2bow(used_news_content[news_id_2][single_feature])
        vec.append(com_simi(news_1_tf, news_2_tf, num_feature))
        news_1_title_tfidf = model[news_1_tf]
        news_2_title_tfidf = model[news_2_tf]
        vec.append(com_simi(news_1_title_tfidf, news_2_title_tfidf, num_feature))
    return vec


def compute_precision(news_id_list, news_dict):
    raw_news_num = len(news_id_list)
    news_event_ = defaultdict(int)
    for single_news in news_id_list:
        news_event_[news_dict[single_news]['specialId']] += 1
    news_event_ = sorted(news_event_.items(), key=lambda c: -c[1])
    precision = news_event_[0][1] / raw_news_num
    # news_event_: [(event_id_1, news_num),(event_id_2, news_num),...]
    return news_event_[0], precision


def second_cluster(first_cluster_list, clf, news_dict):
    # first_cluster_list: single list, include news_id in this community
    print('====================================================================')
    print('the community size of first cluster list is : %d' % len(first_cluster_list))
    if len(first_cluster_list) < 20:
        print('The community_size is too small, will not include any event.')
        return

    used_news_content, corpus = topic_corpus(first_cluster_list, news_dict)

    # use gensim build tfidf model with each topic community
    print('build tf_idf model...')
    dct = Dictionary(corpus)
    corpus2id = [dct.doc2bow(line) for line in corpus]
    tf_idf_model = TfidfModel(corpus2id)

    # build 2-pairs news with topic community
    com_pairs = combinations(list(used_news_content.keys()), r=2)

    # build graph in each topic community
    print('build graph...')
    g = nx.Graph()
    for single_pair in tqdm(list(com_pairs)):
        vec = get_feature_vecture(single_pair[0], single_pair[1], dct, used_news_content, model=tf_idf_model)
        proba = clf.predict_proba([vec])
        if proba[0][1] >= 0.5:
            g.add_edge(single_pair[0], single_pair[1])
    print('the nodes of graph: %d' % g.number_of_nodes())
    print('the egdes of graph: %d' % g.number_of_edges())

    # community detection us GN
    communities_generator = community.girvan_newman(g)
    max_community_num = 40
    print('community detection...the max community size is %d' % max_community_num)
    for single in communities_generator:
        single = sorted(single, key=lambda c: -len(c))
        if len(single[0]) <= max_community_num:
            community_result = single
            break

    # statistics info
    min_event_size = 15
    count_event = 0
    # news_event: [((event_id_1, news_num), precision),   ((event_id_2, news_num), precision)]
    news_event_precision = []
    news_event = []
    true_news = 0
    for single_list in community_result:
        if len(single_list) < min_event_size:
            # the num of news < min_event_size, not a event
            break
        count_event += 1
        news_event_, precision = compute_precision(single_list)
        news_event_precision.append((news_event_, precision))
        news_event.append(news_event_)

    # print info
    print('Detect %d event in this community: ' % count_event)
    for single_news_event in news_event_precision:
        event_, num = single_news_event[0]
        true_news += num
        precision = single_news_event[1]
        print('Event_id: num_news -> %s: %d; precision: %f' % (event_, num, precision))
    #     print('the precision of this community: %f' % (true_news / len(first_cluster_list)))
    return news_event


def run_second_cluster():
    first_cluster_truth, first_cluster_result, news_dict, clf = load_data()
    second_cluster_result = []
    for single_first_cluter_community in first_cluster_result:
        single_first_cluter_community_news_list = single_first_cluter_community[1]
        second_cluster_ = second_cluster(single_first_cluter_community_news_list, clf, news_dict)
        if second_cluster_ is not None:
            second_cluster_result.extend(second_cluster_)


if __name__ == '__main__':
    run_second_cluster()


