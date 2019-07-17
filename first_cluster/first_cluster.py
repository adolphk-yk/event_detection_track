from first_cluster.data_prepare import DataPrepare
from itertools import combinations
import networkx as nx
import sys
import os
import copy
from networkx.algorithms import community
import jieba
from gensim import similarities
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import collections
import pickle


def combination_keywords(keyword):
    if keyword is not None:
        keyword_split = keyword.split(';')
        return list(combinations(keyword_split, r=2))
    return []


def build_pair_dict(data_prepare, co_occurrence_num):
    '''
    build keywords pair_dict
    e.g. {(keyword1, keyword2): num of co-occurrence, ...}
    :return:
    '''
    print('===build_pair_dict===')
    # data_prepare = DataPrepare()
    keyword_pair_dict = {}
    keyword_pair_dict_useful = {}
    for key, value in data_prepare.useful_news_dict.items():
        combination_result = combination_keywords(value['keywords'])
        if len(combination_result) > 0:
            for single_tuple in combination_result:
                if single_tuple in keyword_pair_dict:
                    keyword_pair_dict[single_tuple] += 1
                else:
                    tuple_temp = (single_tuple[1], single_tuple[0])
                    if tuple_temp in keyword_pair_dict:
                        keyword_pair_dict[tuple_temp] += 1
                    else:
                        keyword_pair_dict[single_tuple] = 1

    for key, value in keyword_pair_dict.items():
        if value >= co_occurrence_num:
            keyword_pair_dict_useful[key] = value
    print('The num of keyword_pair_dict items: %d' % len(keyword_pair_dict_useful))
    return keyword_pair_dict_useful


def build_graph(keyword_pair_dict):
    print('===build graph===')
    g = nx.Graph()
    for key in keyword_pair_dict.keys():
        g.add_edge(key[0], key[1])
    print('The node of graph: %d' % len(g.node))
    print('The edge of graph: %d' % len(g.edges))

    return g


def community_detection(graph, max_community_keywords_num, k):
    '''
    use k-clique algorithm for community_detection on keywords graph built above
    :param graph:
    :param max_community_keywords_num:the max keywords num which community has
    :param k: k in k-clique algorithm
    :return: community list(community is represented by keywords set)
    '''
    print('===community detection===')
    max_single_community_size = sys.maxsize
    pre_community_result = len(graph.nodes)
    temp_g = copy.deepcopy(graph)
    k_clique_result_community = []
    epoch_num = 1
    while (max_single_community_size > max_community_keywords_num):
        print('========================================')
        print('community detection epoch: %d' % epoch_num)
        print('k=%d' % k)
        k_clique_community = community.k_clique_communities(temp_g, k)
        k_clique_community = [list(single_community) for single_community in list(k_clique_community)]
        k_clique_community = sorted(k_clique_community, key=lambda c: -len(c))
        # print(k_clique_community[0])
        if len(k_clique_community) > 1:
            delete_node = k_clique_community[1:]
            max_single_community_size = len(k_clique_community[0])
            for delete_list in delete_node:
                k_clique_result_community.append(delete_list)
                temp_g.remove_nodes_from(delete_list)
        if max_single_community_size == pre_community_result:
            k += 1
        pre_community_result = max_single_community_size
        epoch_num += 1
        if len(k_clique_community) == 0:
            break
    if len(k_clique_community) > 0:
        k_clique_result_community.append(k_clique_community[0])
    with open('../temp/first_cluster/community_detection.txt', 'w', encoding='utf-8') as fin:
        for single in k_clique_result_community:
            fin.writelines(';'.join(single) + '\n')

    return k_clique_result_community


def build_dictionary(keyword_pair_dict, dictionary_file_path='../temp/first_cluster/used_keyword.txt'):
    used_keyword_set = set()
    if os.path.isfile(dictionary_file_path):
        with open(dictionary_file_path, 'r', encoding='utf-8') as fin:
            for single in fin:
                used_keyword_set.update(single.replace('\n', ''))
    else:
        for single_pair in keyword_pair_dict.keys():
            used_keyword_set.update(single_pair)
            with open('used_keyword.txt', 'w', encoding='utf-8') as fin:
                for single_keyword in used_keyword_set:
                    fin.write('%s\n' % single_keyword)
    used_keyword_set = list(used_keyword_set)
    return used_keyword_set, dictionary_file_path


def load_stop_words(stop_words_path='../data/stop_words.txt'):
    stop_words = []
    with open(stop_words_path, 'r', encoding='utf-8') as fin:
        for single_word in fin:
            stop_words.append(single_word.replace('\n', ''))
    return stop_words


def first_cluster(data_prepare, dictionary_file_path, stop_words, k_clique_result_community):
    '''
    compute the most similarity community for each news
    :param data_prepare:
    :return:
    '''
    print('===first cluster===')
    # load dictionary
    jieba.load_userdict(dictionary_file_path)
    # use news title and digest to represent news
    corpus_all_after_cut = []
    for key, value in data_prepare.useful_news_dict.items():
        temp_str = value['digest'] + ';' + value['newsTitle']
        temp_cut = []
        for single_cut in jieba.lcut(temp_str):
            if (single_cut not in stop_words) and len(single_cut) > 1:
                temp_cut.append(single_cut)
        corpus_all_after_cut.append(temp_cut)
    corpus_len = len(corpus_all_after_cut)
    for single in k_clique_result_community:
        corpus_all_after_cut.append(single)
    # compute similarity
    dct = Dictionary(corpus_all_after_cut)
    corpus2id = [dct.doc2bow(line) for line in corpus_all_after_cut]
    tf_idf_model = TfidfModel(corpus2id)
    community_corpus = corpus2id[corpus_len:]
    index = similarities.SparseMatrixSimilarity(tf_idf_model[community_corpus], num_features=len(dct.token2id.keys()))
    news_corpus = corpus2id[0:corpus_len]

    result_community2news_id = collections.defaultdict(list)
    for news_id, single_news_corpus in zip(list(data_prepare.useful_news_dict.keys()), news_corpus):
        temp_sim = list(enumerate(index[tf_idf_model[single_news_corpus]]))
        temp_sim = sorted(temp_sim, key=lambda c: -c[1])
        result_community2news_id[temp_sim[0][0]].append(news_id)

    with open('../temp/first_cluster/result_community2news_id.pkl', 'wb') as fout:
        pickle.dump(result_community2news_id, fout)

    result_community2news_id = sorted(result_community2news_id.items(), key=lambda d: d[0])

    with open('../temp/first_cluster/topic_cluster.txt', 'w', encoding='utf-8') as fin:
        for single in result_community2news_id:
            key = single[0]
            value = single[1]
            fin.write('==================Community {0}==================\n'.format(key))
            fin.write('Community keywords: ' + ';'.join(k_clique_result_community[key]) + '\n')
            for single_news_id in value:
                fin.write(data_prepare.useful_news_dict[single_news_id]['newsTitle'] + '\n')


def run_first_cluster():
    data_prepare = DataPrepare()
    keyword_pair_dict = build_pair_dict(data_prepare, co_occurrence_num=3)
    graph = build_graph(keyword_pair_dict)
    k_clique_result_community = community_detection(graph, max_community_keywords_num=15, k=2)
    used_keyword_set, dictionary_file_path = build_dictionary(keyword_pair_dict)
    stop_words = load_stop_words()

    first_cluster(data_prepare, dictionary_file_path, stop_words, k_clique_result_community)


if __name__ == '__main__':
    run_first_cluster()
