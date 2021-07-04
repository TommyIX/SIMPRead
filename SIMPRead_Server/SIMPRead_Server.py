import math
import flask
import spacy
import torch
import itertools

from model_ebldl import *
from train import evaluate
from PreProcess import *

app = flask.Flask(__name__)
# 各项参数设置=======================================================
model_path = './model.pth'
parser = spacy.load('en_core_web_trf')
# =================================================================


def load_best_model (device, model_path = './model.pth', store_dict = "./data"):
    article = Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>', lower=True, tokenizer_language='en_core_web_trf', include_lengths=True)
    summary = Field(tokenize='spacy',init_token='<sos>',eos_token='<eos>',lower=True, tokenizer_language='en_core_web_trf')

    train_data, valid_data, test_data = TabularDataset.splits(path=store_dict, train='train.csv', validation='val.csv', test='test.csv', format='csv', fields=[("text",article),('headline',summary)])

    article.build_vocab(train_data, min_freq=2)
    summary.build_vocab(train_data, min_freq=2)

    _, _, test_loader = BucketIterator.splits((train_data, valid_data, test_data), batch_size=64, sort_within_batch=True, sort_key = lambda x:len(x.text), device=device)

    attention_layer = Attention(enc_hid_dim = 512, dec_hid_dim = 512)
    encode_layer = Encoder(vocab=len(article.vocab),embeding_dim=256, encoder_hidden_dim=512, decoder_hidden_dim=512, dropout=0.5)
    decode_layer = Decoder(output_dim=len(summary.vocab),emb_dim=256, enc_hid_dim=512, dec_hid_dim=512, dropout=0.5, attention=attention_layer)
    model = Seq2Seq(encode_layer,decode_layer, article.vocab.stoi[article.pad_token], device).to(device)

    sum_pad_ids = summary.vocab.stoi[summary.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index = sum_pad_ids)

    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loss = evaluate(model, test_loader, criterion)
    print("Loaded Best Model Info:")
    print(f'Test Loss: {test_loss:.3f} / Test PPL: {math.exp(test_loss):7.3f}')

    return article,summary,model


def predict(sentence, src_field, trg_field, model, device, max_len = 50):
    model.eval()
    if sentence is str:
        nlp = spacy.load('en_core_web_trf')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(src_indexes)]).to(device)
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len.cpu())
    mask = model.create_mask(src_tensor)        
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
        attentions[i] = attention            
        pred_token = output.argmax(1).item()        
        trg_indexes.append(pred_token)
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:], attentions[:len(trg_tokens)-1]

def spacy_tokenizer(sentence):
    tokens = parser(sentence)
    tokens = [tok.lower_ for tok in tokens]
    return tokens

def cleanoutput(prediction):
    def changestarttoupper(string):
        list1 = []
        for i in string: list1.append(i)
        list1[0] = list1[0].upper()
        return "".join(list1)

    curedprediction = [k for k, g in itertools.groupby(prediction)]
    words_delete_index = []   # 存放要删除的重复字符出现的位置
    for pos1 in range(len(curedprediction)):
        pos1_number = curedprediction[pos1+1:].count(curedprediction[pos1])   # 当前字符在每一行中出现的次数
        if pos1_number == 0:   # 如果当前字符在这一行中没有重复，则跳过
            continue
        else:
            pos2 = pos1
            for pos1_repeated_times in range(pos1_number):   # 对比每个重复出现的位置的下一个字符是否也是重复的
                pos2 = curedprediction.index(curedprediction[pos1], pos2+1, len(curedprediction))   # 找到当前字符下一次出现的位置
                if pos2 >= len(curedprediction)-1:   # 如果已经查询到这一行数据的最后一个字符，则跳过
                    continue
                else:
                    if curedprediction[pos1+1] == curedprediction[pos2+1]:   # 判断当前字符的下一个字符是否与重复出现的字符的下一个字符相等
                        words_delete_index.append(pos2)
                        words_delete_index.append(pos2+1)
                    else:
                        continue
    words_delete_index = list(set(words_delete_index))   # 去掉需要重复删除的索引
    words_delete_index.sort(reverse=True)   # 对要删除的位置索引”从大到小“排列，方便后续删除操作
    for delete_index in words_delete_index:
        del curedprediction[delete_index]
    del curedprediction[-1]
    strtoreturn = curedprediction[0]
    for i in curedprediction[1:]:
        if i[0] in [",","'"]:
            strtoreturn+=i
        elif i[0] in [".","\""]:
            i[0] = changestarttoupper(i[0])
            strtoreturn+=i
        else:
            strtoreturn+=(" "+i)
    strtoreturn+="."
    return changestarttoupper(strtoreturn)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("正在加载模型...\nLoading TextSUM model to Device: ", device)
# article,summary,model = load_best_model(device,model_path = model_path ,store_dict="./data")
# print("模型加载成功！ The model is successfully initted!")


@app.route("/predict", methods=["GET","POST"])
def predicttext():
    successfully_handled = False
    string = ""
    try:
        string = flask.request.values['string']
    except:
        pass
    print("Received:", string)
    return "theblyatis:"+string
    # text_tokenized = spacy_tokenizer(text)
    # prediction, _ = predict(text_tokenized, article, summary, model, device)
    # return cleanoutput(prediction)

app.run(host='0.0.0.0',#任何ip都可以访问
        port=8999,#端口
        debug=True)
