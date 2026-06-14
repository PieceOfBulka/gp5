import os
import re
import csv
from dotenv import load_dotenv
import pandas as pd
import logging
from time import time

import lyricsgenius
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

logging.basicConfig(level=logging.INFO, filename='logs/api_collection_logs.txt', encoding="utf-8", filemode='w')

load_dotenv()

genius = lyricsgenius.Genius(os.getenv('GENIUS_TOKEN'),timeout=15,retries=3)
genius.skip_non_songs=True 
genius.remove_section_headers=False 
spotify = Spotify(auth_manager=SpotifyClientCredentials(client_id=os.getenv('CLIENT_ID'),client_secret=os.getenv('CLIENT_SECRET')))


def clean_lyrics(text):
    if not text:
        return ''
    text=re.sub(r'^.*Lyrics','',text)
    text=re.sub(r'You might also like','',text,flags=re.IGNORECASE)
    text=text.strip()
    text=re.sub(r'\d*Embed$','',text)
    return text.strip()

logging.info('=Начало работы скрипта...\n\n')
start_time = time()
data=[]
times=[]
with open('../data/genres_top.csv', 'r', encoding='utf-8') as f:
    f.readline()
    for i in range(100):
        genre,count_value = f.readline().strip().split(',')
        logging.info(f'\n\n--Обработка жанра {genre} {i+1}/100')
        genre_start = time()

        try:
            genre_tracks = spotify.search(f'genre:"{genre}"', type='track', market='US', limit=30)
        except:
            try:
                genre_tracks = spotify.search(f'genre:"{genre}"', type='track', market='US', limit=10)
            except:
                logging.error(f'Не подходит лимит для запроса в спотифай')
                continue

        if 'tracks' in genre_tracks and genre_tracks['tracks']['items']:
            for track in genre_tracks['tracks']['items']:
                song_name = track['name']
                if not song_name:
                    continue
                artists = [artist['name'] for artist in track['artists']]
                if not artists:
                    continue

                genius_res=genius.search_song(song_name,artists[0])
                if not genius_res:
                    continue
                song_text = clean_lyrics(genius_res.lyrics)
                if not song_text:
                    continue
                
                data.append({'song':song_name, 'artists':artists, 'genres':genre, 'lyrics':song_text})
                logging.info(f'Добавлена песня {song_name}')
        
        pd.DataFrame(data).to_csv('../data/additional_songs.csv', index=False, encoding='utf-8', escapechar="\\", quoting=csv.QUOTE_MINIMAL)
        times.append(time()-genre_start)
        logging.info(f'\n--Обработан жанр {genre} за {round(times[-1],2)} секунд')

logging.info('\n\n=Конец работы скрипта...')
logging.info(f'Получили всего новых песен = {len(data)}')
logging.info(f'Общее время выполнения = {round((time()-start_time)/60,2)} минут')
logging.info(f'Среднее время выполнения на жанр = {round(sum(times)/len(times),2)} секунд')