import argparse
import json
from updated_manual_inference import load_static
from manual_inference import predict
from flask import Flask,request
app = Flask(__name__)
##routing
class args:
    model_to_load = '../saved_model/trained_model.pt'
    database='body_builder'
    seed=90
    data_set='spider'
    batch_size=1
    cuda=False
    conceptNet='../data/spider/conceptNet'
    encoder_pretrained_model='bert-base-uncased'
    max_seq_length=512
    column_pointer=True
    embed_size=300
    hidden_size=300
    action_embed_size=128
    att_vec_size=300
    type_embed_size=128
    col_embed_size=300
    readout='identity'
    column_att='affine'
    dropout=0.3
    beam_size=1
    decode_max_time_step=40
    data_dir='../data/spider'
    database_path='../data/spider/original/database/dealPlatform/dealPlatform.db'

args,grammar,model,nlp,tokenizer,related_to_concept,is_a_concept, schemas_raw, schemas_dict = load_static(args)
@app.route('/testing',methods=['GET'])
def testing():
    question = request.args['question']
    result = predict(question)
    data = {}
    data['sql'] = result
    return data

@app.route('/')
def helloIndex():
    return 'Hello World from DashBoard!'


if __name__ == "__main__":
    app.run(debug=True)