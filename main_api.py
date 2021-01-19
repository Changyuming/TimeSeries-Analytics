# web_api及網芳模組
from flask import Flask,render_template,url_for,flash, request, redirect, url_for, jsonify, make_response
import requests
import configparser
import subtraction,threading,webbrowser
import os
import socket
app = Flask(__name__,static_folder='builder', static_url_path="/")
from flask_cors import CORS
#import module
#import json
#import compare_2
#import read_DB

cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["DEBUG"] = True





@app.route('/api/train' , methods=['GET','POST'])
def receive():
    info = request.get_json()
    result = subtraction.main(info)
    result = make_response(result)
    return  result

'''
@app.route('/api/compare/<string:name>' , methods=['GET','POST'])
def receive_2(name):
    info = request.get_json()
    result = compare_2.main_run(name,info)
    return  result
'''
@app.route('/' , methods=['GET','POST'])
def content_hello():
    print("hello")
    return  'Hello !'
'''
@app.route('/DB' , methods=['GET','POST'])
def content_DBoutput():
    print("test")
    info = request.get_json()
    print(info)
    result = read_DB.output_predict(info)
    result = make_response(result)
    return  result
'''

# @app.route('/', defaults={'path': ''})
# @app.route('/<path:path>')
# def catch_all(path):
#     return app.send_static_file("index.html")

if __name__ == '__main__':
    # 获取本机计算机名称
    hostname = socket.gethostname()
    # 获取本机ip
    ip = socket.gethostbyname(hostname)
    port=5001
    url = "http://"+ip+':'+str(port)
    # threading.Timer(1.25, lambda: webbrowser.open(url) ).start()
    app.run(
        host= ip,
        port = port,
        debug=True,
        use_reloader=False,
        threaded=True
    )
