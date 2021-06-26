from flask import Flask
from flask import request 

from flask_cors import CORS , cross_origin

from dotenv import load_dotenv
import base64
import json
import time
import os

# Model Imports
# from model import Model
from model.ann import predict

# Util Imports
import numpy as np

load_dotenv()

app = Flask(__name__)

# App Configurations
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.config['CORS_HEADERS'] = 'Content-Type'

# CORS Configurations 
cors = CORS(
    app,
    origins=[
        "http://localhost:4200",
    ],
)

uploads_dir = 'temp'

try :
    os.makedirs(uploads_dir)
except :
    pass

# Routes
@app.route('/',methods=['POST','GET'])
def indexRoute() :
    data = {
        'data':f'{json.loads(request.data)}' ,
        'args':f'{request.args}'
    }
    return data

@app.route('/getchar',methods=['POST'])
def get() :
    print(f'[+] Request : {request}')
    # return {'character' : 'm'}
    try :
        fName = time.time()
        try :
            file_ = request.files['file']
            # print(file_)
            file_.save(f'./temp/{fName}.jpg')
        except :
            return {'STATUS' : 'INVALID_REQUEST'}
        return {'character':predict(f'./temp/{fName}.jpg')}
    except BaseException as e:
        # raise
        print(f'[!] Error : {e}')
        return {'STATUS' : 'INTERNAL_ERROR'}

@app.route('/getwords',methods=['POST'])
def words() :

    return {'RESPONSE' : wordList }

if __name__ == '__main__':
    app.run()