#!/usr/bin/python
# -*- coding: UTF-8 -*-

import urllib2

request = urllib2.Request('http://finance.eastmoney.com/news/1353,20171211811342728.html')  # 构建一个request请求
response = urllib2.urlopen(request)
print response.read()