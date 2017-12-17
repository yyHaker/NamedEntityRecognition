# -*- coding: utf-8 -*-
"""
使用os.system执行conlleval脚本
"""
import os
eval_script = 'conlleval.pl'
predf = 'temp/pred.test'
scoref = 'temp/score.test'
print('%s < %s > %s' % (eval_script, predf, scoref))
os.system('%s < %s > %s' % (eval_script, predf, scoref))