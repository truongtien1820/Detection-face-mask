
#import flasks libraries
from flask import Flask, render_template
from flask_socketio import SocketIO
import webbrowser
from threading import Timer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    Timer(1,lambda: webbrowser.open_new("http://127.0.0.1:5000/")).start()
    socketio.run(app)
    socketio.close_room(app)
    
    