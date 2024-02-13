from flask import Flask,render_template,request
import pickle
import pandas as pd


app = Flask(__name__)


with open('model_data.pkl', 'rb') as model_file:
    model_data = pickle.load(model_file)

model = model_data['model']
label_encoders = model_data['label_encoders']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        artists = request.form['artists']
        album_name = request.form['album_name']
        track_name = request.form['track_name']
        duration_ms = int(request.form['duration_ms'])
        danceability = float(request.form['danceability'])
        energy = float(request.form['energy'])
        key = int(request.form['key'])
        speechiness = float(request.form['speechiness'])
        instrumentalness = float(request.form['instrumentalness'])
        liveness = float(request.form['liveness'])
        valence = float(request.form['valence'])
        track_genre = request.form['track_genre']

        input_data = [[artists, album_name, track_name, duration_ms, danceability,
                       
        energy, key, speechiness, instrumentalness, liveness, valence, track_genre]]

        columns = ['artists', 'album_name', 'track_name', 'duration_ms', 'danceability',
       'energy', 'key', 'speechiness', 'instrumentalness', 'liveness', 'valence', 'track_genre']

        dtype_dict = {'duration_ms': int , 'danceability': float, 'energy': float , 'key': int,
                      'speechiness': float, 'instrumentalness': float , 'liveness': float ,
                      'valence': float}

        new_df = pd.DataFrame(input_data, columns=columns)

        new_df = new_df.astype(dtype_dict)

        artists_encoder = label_encoders['artists']
        album_name_encoder = label_encoders['album_name']
        track_name_encoder = label_encoders['track_name']
        track_genre_encoder = label_encoders['track_genre']

        # Transform individual columns using their respective encoders
        new_df['artists'] = artists_encoder.transform(new_df['artists'])
        new_df['album_name'] = album_name_encoder.transform(new_df['album_name'])
        new_df['track_name'] = track_name_encoder.transform(new_df['track_name'])
        new_df['track_genre'] = track_genre_encoder.transform(new_df['track_genre'])


        prediction = model.predict(new_df)

        return render_template('index.html', prediction=str(prediction[0]))


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)






