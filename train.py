#-*- coding:utf8 -*-
import logging
import fasttext
import jieba
# help(fasttext.fasttext)
# exit()
input="清华大学"

text = jieba.cut_for_search(input)
text = " ".join(text)

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

classifier = fasttext.train_supervised("data/trained.txt",epoch=10,ws=2 , lr=0.5, wordNgrams=1, dim=1000,label="__label__",loss='softmax')

classifier.save_model("model/classify.model")

result = classifier.predict(text)

print("预测词：" + input + "\n")

print("预测结果：")
print( result)