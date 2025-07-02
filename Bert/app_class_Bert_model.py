from flask import  render_template
from flask import send_file
import torch
import os
import numpy as np
from flask import Flask
from flask import request
from model import BERT_Arch
from transformers import BertModel, BertTokenizer
from flask.views import MethodView
import time 
app = Flask(__name__)

class ActionItemsApI(MethodView):

    def __init__(self):
        path = 'Multiligual_BERT_128_Urdu_V.1.4.pt'
        bert = BertModel.from_pretrained("bert-base-multilingual-uncased")
        MODEL = BERT_Arch(bert)
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.DEVICE)
        # self.MODEL = nn.DataParallel(MODEL)
        MODEL.load_state_dict(torch.load(path, map_location=self.DEVICE))
        MODEL.to(self.DEVICE)
        self.predictor=MODEL.eval()
        self.TOKENIZER = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")

    def sentence_prediction(self,sentence):
        tokenizer = self.TOKENIZER
        max_len = 128
        tokens = tokenizer.tokenize(sentence)
        tokens = tokens[:max_len - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = input_ids + [0] * (max_len - len(input_ids))
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        input_ids = input_ids.to(self.DEVICE)

        input_mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
        input_mask = torch.tensor(input_mask).unsqueeze(0)
        input_mask = input_mask.to(self.DEVICE)
        outputs = self.predictor(
            sent_id=input_ids,
            mask=input_mask,
        )
        #print("outputs",outputs)
        outputs = torch.sigmoid(outputs).cpu().detach().numpy()
        preds = np.argmax(outputs, axis=1)
        #print("origional",preds)
        return preds

    def action_items(self):
        if request.method == "POST":
            file = request.files["file"]
            if file and file.content_type == 'text/plain':
                #print('soooo')
                try:
                    start_time = time.perf_counter()
                    f_name = file.filename
                    file_name = os.path.basename(f_name.replace('\\',os.sep))
                    file_name = os.path.join("/home/cle-dl-11/action_items/recieved_text_files/", file_name)

                    file.save(file_name)
                    with open(file_name,'r') as re:
                        content = re.readlines()
                    #prediction = []
                    result_dict = {}
                    for item in content:
                        item = item.strip()
                        pred = self.sentence_prediction(item)
                        result_dict[item] = pred[0]
                        #prediction.append(self.sentence_prediction(item))

                    #pred = self.summarization_prediction(content)
                    #message = prediction
                    with open("action_item.txt", "w") as f:
                        for item in result_dict:
                            pree = result_dict[item]
                            if pree == 1 and len(str(item))>20:
                                print(item,pree)
                                f.write("%s\n" % ( str(item) ))

                    end_time = time.perf_counter()
                    print("All ok ")
                    print(f"Time take by Action Item Api : {end_time - start_time}")
                    return send_file("action_item.txt", as_attachment=True)
                except:
                    return "Something Wents Wrong"
            else:
                return "Send Text file only"
        else:
            return "Please check your file and sent again"
        #model = self.load_Model('saved_weights1.pt')

#will be deleted
    def get(self):
        return render_template("index.html")

    def post(self):
        # model = self.load_Model('saved_weights1.pt')
        senetence = [x for x in request.form.values()]
        prediction = []
        for item in senetence:
            prediction.append(self.sentence_prediction(item))
        return render_template("index.html", prediction_text="The given sentence is  {}".format(str(prediction)))

app.add_url_rule('/', view_func=ActionItemsApI.as_view(name='index'))


#rema
api = ActionItemsApI()
@app.route('/action', methods=["GET", "POST"])
def action():
    return api.action_items()

if __name__ == "__main__":
    app.run(debug=True)

