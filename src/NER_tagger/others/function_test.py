# -*- coding: utf-8 -*-
from src.NER_tagger.loader import *


sentences = load_sentences('./dataset/eng.train', None, False)
for sentence in sentences:
    print(sentence)