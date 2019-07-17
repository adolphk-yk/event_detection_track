from collections import defaultdict
import pickle
import os


def load_news(news_path='./data/text_keywords.csv'):
    print('===load news data===')
    text_keywords = []
    id = 0
    sample = ''
    with open(news_path, 'r', encoding='utf-8') as fin:
        temp = fin.readlines()

    for index, single_line in enumerate(temp):
        if id == 0:
            text_keywords.append(single_line)
            id += 1
        else:
            if single_line.find(str(id)) == 0:
                text_keywords.append(sample)
                id += 1
                sample = ''
            single_line = single_line.replace('\\n', '').replace('\n', '').replace('\\t', '')
            sample = sample + single_line
            if index == len(temp) - 1:
                text_keywords.append(sample)
    if '' in text_keywords:
        text_keywords.remove('')

    text_keywords_title = text_keywords[0].replace('\n', '').split('\t')
    text_keywords_col_num = len(text_keywords_title)
    print(text_keywords_title)
    del text_keywords[0]
    print('news num:', len(text_keywords))
    news_dict = defaultdict(dict)
    illegal = []
    for single_news in text_keywords:
        single_news_split = single_news.split('\t')
        if len(single_news_split) == text_keywords_col_num:
            news_dict[single_news_split[1]] = {'newsTitle': single_news_split[2],
                                               'newsContent': single_news_split[3],
                                               'keywords': single_news_split[4],
                                               'publishDate': single_news_split[5]}
        else:
            illegal.append(single_news)
    print('legal news line:', len(news_dict))
    print('illegal new lines:', len(illegal))
    return news_dict


def load_event(event_path='./data/event_sample_special.csv'):
    with open(event_path, 'r', encoding='utf-8') as fin:
        event_sample_special = fin.readlines()
    event_sample_special_title = event_sample_special[0].replace('\n', '').split('\t')
    event_sample_special_col_num = len(event_sample_special_title)
    del event_sample_special[0]
    print('event_sample_special length:', len(event_sample_special))
    print('the col of event_sample_special:', event_sample_special_title)
    event_dict = defaultdict(dict)
    illegal = 0
    for single_line in event_sample_special:
        single_line_split = single_line.split('\t')
        if len(single_line_split) == event_sample_special_col_num:
            event_dict[single_line_split[2]] = {'channelName': single_line_split[1],
                                                'specialTitle': single_line_split[3],
                                                'specialPublishDate': single_line_split[4]}
        else:
            illegal += 1
    print('legal line num:', len(event_dict))
    print('illegal line num:', illegal)
    return event_dict


def process_data(event_id_path='./data/event_sample_content_id.txt',
                 event_context='./data/event_sample_content.csv'):
    event_sample_content = []
    event_sample_content_id = []
    id = 0
    sample = ''

    with open(event_id_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            event_sample_content_id.append(line.replace('\n', ''))
    del event_sample_content_id[0]

    with open(event_context, 'r', encoding='utf-8') as fin:
        temp = fin.readlines()

    for index, single_line in enumerate(temp):
        if id == 0:
            event_sample_content.append(single_line)
            id += 1
        else:
            if id < len(event_sample_content_id) and single_line.find(event_sample_content_id[id]) == 0:
                event_sample_content.append(sample)
                id += 1
                sample = ''
            single_line = single_line.replace('\\n', '').replace('\n', '').replace('\\t', '')
            sample = sample + single_line
            if index == len(temp) - 1:
                event_sample_content.append(sample)
    del event_sample_content[0]
    if '' in event_sample_content:
        event_sample_content.remove('')
    event_sample_content_title = event_sample_content[0].replace('\n', '').split('\t')
    event_sample_content_col_num = len(event_sample_content_title)
    print('File title :', event_sample_content_title)
    print('event_sample_content length:', len(event_sample_content))

    illegal = []
    event_news_dict = defaultdict(dict)
    news_event_dict = defaultdict(dict)
    for single_sample in event_sample_content:
        single_sample_split = single_sample.split('\t')
        if len(single_sample_split) == event_sample_content_col_num:
            news_event_dict[single_sample_split[4]] = {'specialId': single_sample_split[2],
                                                       'content': single_sample_split[7],
                                                       'digest': single_sample_split[9],
                                                       'keywords': single_sample_split[8],
                                                       'newsTitle': single_sample_split[5]}
            if len(event_news_dict[single_sample_split[2]]) == 0:
                event_news_dict[single_sample_split[2]] = {'channelName': single_sample_split[1],
                                                           'specialTitle': single_sample_split[3],
                                                           'news': list([single_sample_split[4]])}
            elif len(event_news_dict[single_sample_split[2]]) > 0:
                event_news_dict[single_sample_split[2]]['news'].append(single_sample_split[4])
        else:
            illegal.append(single_sample)

    return event_news_dict, news_event_dict


def filter_data(event_news_dict, news_event_dict, news_dict, event_save_path='./temp/event_news.pkl', news_save_path='./temp/news.pkl'):
    for key, value in event_news_dict.items():
        temp = []
        for single_event in value['news']:
            if len(news_event_dict[single_event]['keywords']) > 0 or (
                    len(news_dict[single_event]) > 0 and len(news_dict[single_event]['keywords']) > 0):
                temp.append(single_event)
        if len(temp) == 0:
            event_news_dict[key] = {}
        else:
            event_news_dict[key]['news'] = temp

    # save useful event and news
    result_news_dict = {}
    result_event_news_dict = {}
    for key, value in event_news_dict.items():
        if len(value) > 0:
            result_event_news_dict[key] = value
            temp_news = value['news']
            for single_news in temp_news:
                single_news_value = news_event_dict[single_news]
                if len(single_news_value['keywords']) == 0:
                    single_news_value['keywords'] = ';'.join(news_dict[single_news]['keywords'].split(','))
                result_news_dict[single_news] = single_news_value

    with open(event_save_path, 'wb') as fout:
        pickle.dump(result_event_news_dict, fout)
    with open(news_save_path, 'wb') as fout:
        pickle.dump(result_news_dict, fout)


def run_process():
    news_dict = load_news()
    event_news_dict, news_event_dict = process_data()
    filter_data(event_news_dict, news_event_dict, news_dict)


def prepare_dir(output_dir='./temp'):
    dir_list = [output_dir]
    cluster_dir = ['first_cluster', 'second_cluster']
    for single_cluster in cluster_dir:
        dir_list.append('/'.join([output_dir, single_cluster]))
    for single in dir_list:
        if not os.path.exists(single):
            os.mkdir(single)


if __name__ == '__main__':
    prepare_dir()
    run_process()