from first_cluster.data_prepare import DataPrepare
import copy
from itertools import combinations


def filter_event(data_prepare):
    pick_event_news_dict = copy.deepcopy(data_prepare.event_news_dict)
    # delete event which has only one news
    for single_event in list(pick_event_news_dict.keys()):
        if len(pick_event_news_dict[single_event]['news']) <= 1:
            del pick_event_news_dict[single_event]
    # delete news which used in first cluster
    for key, value in data_prepare.useful_event_news_dict.items():
        for single_news in value['news']:
            pick_event_news_dict[key]['news'].remove(single_news)
            if len(pick_event_news_dict[key]['news']) == 0:
                del pick_event_news_dict[key]

    pick_event_news_list = []
    for value in pick_event_news_dict.values():
        pick_event_news_list.append(value['news'])

    return pick_event_news_list


def build_neg_sample(single_news, news_list):
    result = []
    for single_list in news_list:
        for single_news_id in single_list:
            result.append((single_news, single_news_id))
    return result


def build_sample(pick_event_news_list):
    '''
    pos_sample comes from same event
    neg_sample comes from different event
    :param pick_event_news_list:
    :return:
    '''
    print('===build sample===')
    neg_sample_set = []
    pos_sample_set = []
    for index, single_list_news in enumerate(pick_event_news_list):
        for single_tuple in combinations(single_list_news, r=2):
            pos_sample_set.append(single_tuple)
        if index + 1 != len(pick_event_news_list):
            for single_news_id in single_list_news:
                neg_sample_set.extend(build_neg_sample(single_news_id, pick_event_news_list[index + 1:]))
    print('have %d postive sample data.' % len(pos_sample_set))
    print('have %d negtive sample data.' % len(neg_sample_set))
    # save as txt file
    with open('../temp/second_cluster/positive.txt', 'w', encoding='utf-8') as fin:
        for single in pos_sample_set:
            fin.write(' '.join(single) + '\n')
    with open('../temp/second_cluster/negative.txt', 'w', encoding='utf-8') as fin:
        for single in neg_sample_set:
            fin.write(' '.join(single) + '\n')


def run_build_sample():
    data_prepare = DataPrepare()
    pick_event_news_list = filter_event(data_prepare)
    build_sample(pick_event_news_list)

if __name__ == '__main__':
    run_build_sample()