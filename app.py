from flask import Flask
from flask import request 

from flask_cors import CORS , cross_origin

from dotenv import load_dotenv
import json
import os

# Model Imports
# from model import Model

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
    file_ = request.files['image'].read()
    f = np.fromstring(file_,np.uint8)
    img = cv2.imdecode(f,cv2.IMREAD_COLOR)
    return {}


if __name__ == '__main__':
    app.run()