import jieba
import fasttext

# input="石碣第十一中学"
# input = "北京育才学校"
input = "我今天就是去北京大学"
# input = "胜和塘贝村村北十二巷1-103号"

text = jieba.cut_for_search(input)
text = " ".join(text)

classify = fasttext.load_model("model/classify.model")
result = classify.predict(text)

print("预测词：" + input + "\n")

print("预测结果：")
print(result)