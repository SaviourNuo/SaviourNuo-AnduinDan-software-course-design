# -- coding:UTF-8 --
import os,sys,time,threading
from PyQt5.QtCore import *
import requests
from bs4 import BeautifulSoup  #网页解析，获取数据
import re #正则表达式，进行文字匹配
import urllib.request  #制定URL，获取网页数据
import json
import GlobalVariable as glovar
'''
思路：获取网址
      获取图片地址
      爬取图片并保存llllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll
'''

# disguise ourself
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/93.0.4577.63 Safari/537.36 '
}  

NUM_PER_GET = 35    # 每次抓取图片数量(35是此网页每次翻页请求数量)
MIN_IMAGE_SIZE = 10  # 抓取图片最小大小(单位字节)，小于此值抛弃

# private函数
def _get_single_image(url, path, imageName):
    try:
        contextMgr = urllib.request.urlopen(url, timeout=5)
        content = contextMgr.read()
        if sys.getsizeof(content) < MIN_IMAGE_SIZE:
            return -1
    except Exception as e:
        # print(url, e)
        return -2
    #提取图片格式
    suffix = url[url.rfind('.'):] # 取出图片的后缀
    #pattern = re.compile("^\\.[a-zA-Z]+")
    # pattern = re.compile(r'http[\S]*jpg')
    pattern = re.compile(".[Jj][Pp][Gg]") # 匹配仅jpg格式的图片
    substr = pattern.match(suffix);
    if not substr: 
        print("not JPG")
        return -4
    suffix  = pattern.match(suffix).group(0) # 匹配正则表达式整体结果

    try:
        if not os.path.exists(path):
            os.mkdir(path)
        f = open(os.path.join(path, imageName+suffix), 'wb')
        f.write(content)
        f.close()
    except Exception as e:
        print(os.path.join(path, imageName+suffix), e)
        return -3
        
    return 0

#private函数
# 获取图片地址并保存下载
def _get_image_set(htmlUrl,keyword,rootPath = "./",totalNeedNum = 20):
    ses = requests.Session() # 创建一个会话
    first = 0
    count = 0
    glovar.set_value("currentGetNum",count)  #写入共享变量表，做线程间通信
    while count < totalNeedNum:
        fullHtmlUrl = htmlUrl%(keyword,first,NUM_PER_GET,NUM_PER_GET)
        read = ses.get(url = fullHtmlUrl, timeout=(3.05, 10) ,headers = headers)
        soup = BeautifulSoup(read.text, "html.parser")
        imgTagList = soup.find_all("a", class_ = "iusc")  # 大概知道是个按CSS属性的匹配，但不太清楚

        for imgTag in imgTagList:
            if count == totalNeedNum: # 这个处理不确定要不要加上
                break # return False # 到达指定数量了
            urldict = json.loads(imgTag.get('m'))  # 这个是最不明白的地方
            if _get_single_image(urldict["murl"], rootPath, "img_{0:05}".format(count)) < 0:
                continue
            count=count+1
            print("Downloaded {0} picture".format(count))
            sys.stdout.flush()
            glovar.set_value("currentGetNum",count)  #写入共享变量表，做线程间通信
            time.sleep(0.01)
        first = first + NUM_PER_GET
        time.sleep(0.1)
    return True
 
# # 主函数 单独调试用
if __name__ == '__main__':
    keywords = ["街道","足球","机械","果蔬","医药","动漫","招牌","海洋","宇宙","肖像"]  #模拟人输入的关键字
    totalNeedNum = 100
    htmlUrl= \
        "https://cn.bing.com/images/async?q=%s&first=%d&count=%d&relp=%d&lostate=r&mmasync=1"
        # %(keywords[0], 0, NUM_PER_GET, NUM_PER_GET)
    # print(htmlUrl)
    
    for keyword in keywords:
        rootPath = "./network_spider/getImage2_test/{0}/".format(keyword)  #保存的根路径
        _get_image_set(htmlUrl,keyword,rootPath,totalNeedNum)

def _getImage(keyword,totalNeedNum = 20,rootPath = "./network_spider/getImage2_test/"):    
    htmlUrl= "https://cn.bing.com/images/async?q=%s&first=%d&count=%d&relp=%d&lostate=r&mmasync=1"
    #保存的根路径
    # if not len(keyword): #字符串判空
    #     print("void keyword")
    #     return -1
    # else :
    if not os.path.exists(rootPath):
        os.mkdir(rootPath)
    print("download start")
    _get_image_set(htmlUrl,keyword,os.path.join(rootPath,keyword),totalNeedNum)
    print("download end")
    return 0


# 爬虫线程类
class SpiderThread(QThread):
    def __init__(self,keyword,totalNeed,storagePath):
        super(SpiderThread, self).__init__()
        # assert len(keyword)>0 #字符串判空
        self.keyword = keyword 
        self.totalNeedNum = 20 if not totalNeed else int(totalNeed) #如果不输入参数则默认爬20张
        self.storagePath = storagePath
        glovar.set_value("totalNeedNum",self.totalNeedNum)
   
    def run(self): 
        _getImage(self.keyword,self.totalNeedNum,self.storagePath)
