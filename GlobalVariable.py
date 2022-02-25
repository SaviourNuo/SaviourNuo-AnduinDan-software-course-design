# -*- coding: utf-8 -*-
 
def _init():#初始化
    global _global_dict
    _global_dict = {}
 
def set_value(key,value):
    """ 定义一个全局变量 """
    _global_dict[key] = value
 
 
def get_value(key):
    """ 获得一个全局变量,不存在则报错 """
    try:
        # print("{0}:{1}".format(key,_global_dict[key]))
        return _global_dict[key]
    except KeyError:
        print("value not found")
        return None