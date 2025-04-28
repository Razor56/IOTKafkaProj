from flask import Flask, send_from_directory

app = Flask(__name__)

@app.route('/')
def serve_html():
    return send_from_directory('.', 'webserver.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, ssl_context=('localhost.pem', 'localhost-key.pem'), debug=True)
# This is a simple Flask server that serves an HTML file.