from flask import  render_template
from flask import send_file
import os
import pandas as pd
import joblib
from flask import Flask
from flask import request
from sklearn.feature_extraction.text import TfidfVectorizer
from flask.views import MethodView

app = Flask(__name__)

class ActionItemsApI(MethodView):

    def __init__(self):
        Model_path = 'logistic_regression_model.pkl'
        self.MODEL = joblib.load(Model_path)
        # Load the saved vectorizer
        tfidf_vectorizer = "tfidf_vectorizer.pkl"
        self.loaded_vectorizer = joblib.load(tfidf_vectorizer)


    def sentence_prediction(self,sentence):
        max_feature_num = 50000
        sentence_series = pd.Series(sentence)
        test_vecs = TfidfVectorizer(max_features=max_feature_num,
                                    vocabulary=self.loaded_vectorizer.vocabulary_).fit_transform(
            sentence_series)
        preds = self.MODEL.predict(test_vecs)
        print("origional",preds)
        return preds

    def action_items(self):
        if request.method == "POST":
            file = request.files["file"]
            if file and file.content_type == 'text/plain':
                #print('soooo')
                try:
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
                        result_dict[item] = pred

                    with open("action_item.txt", "w") as f:
                        for item in result_dict:
                            pree = result_dict[item]
                            if pree == 1 and len(item)>=18:
                                print(len(item),item,pree)
                                f.write("%s\n" % ( str(item) ))

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
            print(item)
            prediction.append(self.sentence_prediction(item))
        return render_template("index.html", prediction_text="The given sentence is  {}".format(str(prediction)))

app.add_url_rule('/', view_func=ActionItemsApI.as_view(name='index'))


#rema
api = ActionItemsApI()
@app.route('/action', methods=["GET", "POST"])
def action():
    return api.action_items()

if __name__ == "__main__":
    print("Okkkkkkkkkkkkkkkkk")
    app.run(debug=True)

