#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : logger.py
# @Time    : 18-3-14
# @Author  : J.W.
import logging

import config

'''
logger 工具类
'''


def contains_this_handler(handlers, handler):
    '''
    判断 日志中是否已经存在该handler
    :param handlers:
    :param handler:
    :return:
    '''
    for hd in handlers:
        if isinstance(hd, handler):
            return True
    return False


log_file = config.log_file
logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)s %(asctime)s %(name)-6s: %(filename)s[line:%(lineno)d]  %(message)s',
                    filename=log_file,
                    filemode='a')  # or 'w', default 'a'

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s %(asctime)s %(name)-6s: %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logger = logging.getLogger("app")

# # 添加TimedRotatingFileHandler
# # 添加后app.log中会重复输出 而且没有时间
# # 根据 when 定义切割方式  midnight M
# filehandler = logging.handlers.TimedRotatingFileHandler(log_file, when='midnight', interval=1,
#                                                         backupCount=0)
# # 设置后缀名称，跟strftime的格式一样
# filehandler.suffix = "%Y-%m-%d.log"
# logger.addHandler(filehandler)
