from flask import Flask, request
from flask_cors import CORS, cross_origin
import service

# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)
cors = CORS(app) # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'


# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return 'Hello World'

@app.route('/ticker')
@cross_origin()
def return_all_tickers():
    return service.get_all_tickers()

@app.route('/sentiment') 
@cross_origin()
def query_example(): 
    ticker = request.args.get('ticker') 
    return service.get_sentiment_through_news(ticker=ticker)

# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(debug=True)