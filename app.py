import csv

from flask import Flask, request, jsonify, Response
import pandas as pd
import json
from flask_cors import *
from data import data_preprocess
from main_informer import informer
from othermodels import simple_exp,holt
import run_LSTM
from data import figure

app = Flask(__name__)
CORS(app, supports_credentials=True)
data_file_path = "data/bridge1.csv"


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/login', methods=['POST'])
def login():
    recv_data = request.get_data()
    json_re = json.loads(recv_data)
    users = {}
    with open('user.txt', 'r') as file:
        for content in file.readlines():
            print(content)
            content = content.strip('\n').split(' ')
            print(content)
            users[content[0]] = content[1]
    if users.get(json_re['Id']) == json_re['Password']:
        return jsonify({'ok': True})
    return jsonify({'ok': False})


@app.route('/register', methods=['POST'])
def register():
    recv_data = request.get_data()
    json_w = json.loads(recv_data)
    id = json_w.get('Id')
    with open('user.txt', 'r') as filea:
        for content in filea.readlines():
            content = content.strip('\n').split(' ')
            if content[0] == id:
                return jsonify({'ok': False})
    with open('user.txt', 'a') as file:
        file.write('\n')
        id = json_w.get('Id')
        password = json_w.get('Password')
        file.write(id)
        file.write(' ')
        file.write(password)
    return jsonify({'ok': True})


@app.route('/data', methods=['GET'])
def get_data():
    recv_date = request.values.get("date")
    bridge_data = pd.read_csv(data_file_path)
    if recv_date == '':
        data = [{'date': str(row['date']), 'y': row['y']} for _, row in bridge_data.iterrows()]
    else:
        data = [{'date': str(row['date']), 'y': row['y']} for _, row in bridge_data.iterrows() if str(row['date']).find(str(recv_date)) != -1]
    return json.dumps({'data': data})


@app.route('/data', methods=['POST'])
def edit_data():
    recv_data = request.get_data()
    try:
        bridge_data = pd.read_csv(data_file_path)
        json_re = json.loads(recv_data)
        bridge_data = bridge_data.append([json_re])
        bridge_data.to_csv(data_file_path, index=False, encoding='utf-8')
        return jsonify({'result': 0})
    except:
        return jsonify({'result': 1})


@app.route('/dataClean', methods=['POST'])
def data_clean_pro():
    data_preprocess.data_clean()
    return json.dumps({'success': True})


@app.route('/dataClean', methods=['GET'])
def dataClean():
    recv_date = request.values.get("date")
    bridge_data = pd.read_csv('data/bridge111.csv')
    if recv_date == '':
        data = [{'date': str(row['date']), 'y': row['y']} for _, row in bridge_data.iterrows()]
    else:
        data = [{'date': str(row['date']), 'y': row['y']} for _, row in bridge_data.iterrows() if str(row['date']).find(str(recv_date)) != -1]

    return json.dumps({'data': data})


@app.route('/data/img', methods=['GET'])
def dataClean_img():
    recv_date = request.values.get("clean")
    if recv_date == "true":
        recv_date = True
    else:
        recv_date = False
    figure.show_figure(recv_date)
    img_path = "data.png"
    with open(img_path, 'rb') as f:
        image = f.read()
    return Response(image)


@app.route('/predict', methods=['GET'])
def get_predict_result():
    recv_date = request.values.get("model")
    clean = request.values.get("clean")
    img_path = "img.png"
    # todo
    # 调用模型，open 对应图片地址
    if clean == "true":
        clean = True
    else:
        clean = False
    if recv_date == "Simple_Exp":
        simple_exp.simple_exp(clean)
        img_path = "simple_exp.png"
    elif recv_date == "Holt":
        holt.holt(clean)
        img_path = "holt.png"
    elif recv_date == "LSTM":
        run_LSTM.LSTM(clean)
        img_path = "lstm.png"
    elif recv_date == "Informer":
        informer(clean)
        img_path = "informer.png"
    with open(img_path, 'rb') as f:
        image = f.read()
    return Response(image)


if __name__ == '__main__':
    app.run(debug=True)
