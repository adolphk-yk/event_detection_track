import pickle


class DataPrepare:
    '''
    load data from file data process results.
    filter event which has too less or many news.
    '''
    def __init__(self, max_news_num=100, min_news_num=20):
        self.event_news_path = '../temp/event_news.pkl'
        self.news_path = '../temp/news.pkl'
        self.event_news_dict = {}
        self.news_dict = {}
        self.useful_event_news_dict = {}
        self.useful_news_dict = {}
        self.load_data()
        self.filter_data(min_news_num=min_news_num, max_news_num=max_news_num)

    def load_data(self):
        print('===data load===')
        with open(self.event_news_path, 'rb') as fout:
            self.event_news_dict = pickle.load(fout)
        print('there are %d event in dataset.' % len(self.event_news_dict))
        with open(self.news_path, 'rb') as fout:
            self.news_dict = pickle.load(fout)
        print('there are %d news in dataset.' % len(self.news_dict))

    def filter_data(self, min_news_num, max_news_num):
        for key, value in self.event_news_dict.items():
            try:
                news_list = value['news']
                if len(news_list) > min_news_num:
                    if len(news_list) > max_news_num:
                        news_list = news_list[0:100]
                        value['news'] = news_list
                    self.useful_event_news_dict[key] = value
            except:
                pass

        with open('../temp/first_cluster/used_event_news.pkl', 'wb') as fout:
            pickle.dump(self.useful_event_news_dict, fout)

        for value in self.useful_event_news_dict.values():
            for single_news_id in value['news']:
                self.useful_news_dict[single_news_id] = self.news_dict[single_news_id]

        print('the num of event is %d after filter' % len(self.useful_event_news_dict))
        print('the num of news is %d after filter' % len(self.useful_news_dict))