# event_detection_track
Detect and track event from news

程序分为三部分：预处理/第一层聚类/第二层聚类；

在运行任何程序之前请务必保证运行过根目录下的预处理程序(process.py):
```bash
    #when you use this code at the first time, insure to run process.py
    python process.py
```
预处理程序会对数据进行清洗封装，产生event_news.pkl & news.pkl，存放在./temp目录下

其中第二层聚类需要用到第一层的结果.

- first cluster
```bash
    cd first_cluster
    python first_cluster.py
```
第一层聚类后产生的结果存放于 ./temp/first_cluster

- second cluster
```bash
    cd second_cluster
    # if has not run build_sample.py; after run this will generate negative.txt & positiive.txt in ./temp/second_cluster
    python build_sample.py
    # if has not run classifier.py; Run classifier will train a svm classifier and save model at ./temp/second_cluster
    python classifier.py
    
    run second_cluster.py
```