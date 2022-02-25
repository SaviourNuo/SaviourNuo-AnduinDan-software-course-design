# -- coding:UTF-8 --
import os,sys,time
sys.path.append(".")
import urllib,requests,re
from PyQt5.QtCore import *
import GlobalVariable as glovar

# disguise ourself
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/93.0.4577.63 Safari/537.36 '
}
 
NUM_PER_GET = 30    # 每次抓取图片数量(35是此网页每次翻页请求数量)
MIN_IMAGE_SIZE = 10  # 抓取图片最小大小(单位字节)，小于此值抛弃

def _get_page(keyword,page,n):
    """返回带页码的url"""
    page=page*n
    keyword=urllib.parse.quote(keyword, safe='/')
    url_begin= "http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word="
    url = url_begin+ keyword + "&pn=" +str(page) + "&gsm="+str(hex(page))+"&ct=&ic=0&lm=-1&width=0&height=0"
    return url
 
def _get_onepage_urls(onePageUrl):
    """获得一个page里所有img的urlList"""
    try:
        html = requests.get(url=onePageUrl,headers=headers).text
    except Exception as e:
        print(e)
        picUrlList = []
        return picUrlList
    picUrlList = re.findall('"objURL":"(.*?)",', html, re.S)
    return picUrlList

def _get_url_list(keyword,totalNeedNum):
    """获得所有页表里的url，并生成为url列表"""
    page1stNum=0  
    pageMaxNum=int(totalNeedNum/NUM_PER_GET)+1
    urlList = []
    while page1stNum<=pageMaxNum:
        print("{0} request for page".format(page1stNum)) 
        url=_get_page(keyword,page1stNum,NUM_PER_GET) #请求页数
        onepage_urls= _get_onepage_urls(url) #得到当页里的url表
        page1stNum += 1
        urlList.extend(onepage_urls) # url表增添
    return urlList
 
 
def _get_single_image(url, path, imageName):
    try:
        contextMgr = urllib.request.urlopen(url, timeout=15)
        content = contextMgr.read()
        if sys.getsizeof(content) < MIN_IMAGE_SIZE:
            return -1
    except Exception as e:
        return -2# print(url, e)
    suffix = ".jpg"  # 不是很理解为啥百度的picUrl的后缀不是jpg，但又确实是jfif的pic
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

def _get_image_set(urlList,totalNeedNum = 20,rootPath = "./"):
    """给出图片链接列表, 根据url表下载指定数量的图片"""
    count = 0 # 记录当前已下载图片的数量
    for picUrl in urlList:
        if count >= totalNeedNum: 
            break # return False # 到达指定数量了
        if not os.path.exists(rootPath):
            os.mkdir(rootPath)
        picName = "img_{0:05}".format(count)
        if _get_single_image(picUrl,rootPath,picName) <0:
            continue  # 当前的pic下载出错
        count+=1
        print("Downloaded {0} picture".format(count))
        sys.stdout.flush()
        glovar.set_value("currentGetNum",count)  #写入共享变量表，做线程间通信
        time.sleep(0.01)
    return True

def _getImage(keyword="甘雨",totalNeedNum = 20,rootPath = "./network_spider/getImage2_test/"):    
    allPicUrlList = _get_url_list(keyword,totalNeedNum)

    if not os.path.exists(rootPath):
        os.mkdir(rootPath)
    print("download start")
    #print(allPicUrlList)
    _get_image_set(list(set(allPicUrlList)),totalNeedNum,os.path.join(rootPath,keyword))
    print("download end")

    return 0


if __name__ == '__main__':
    keywords=["甘雨",]  
    totalNeedNum = 100
    
    for keyword in keywords:
        rootPath = "./network_spider/getImage2_test/"
        _getImage(keyword,totalNeedNum,rootPath)

# 爬虫线程类
class SpiderThread(QThread):
    def __init__(self,keyword,totalNeed,storagePath):
        super(SpiderThread, self).__init__()
        # assert len(keyword)>0 #字符串判空
        self.keyword = keyword 
        self.totalNeedNum = 20 if not totalNeed else int(totalNeed) #如果不输入参数则默认爬20张
        self.storagePath = storagePath
        glovar.set_value("totalNeedNum",self.totalNeedNum)
        glovar.set_value("currentGetNum",0)
    def run(self): 
        _getImage(self.keyword,self.totalNeedNum,self.storagePath)