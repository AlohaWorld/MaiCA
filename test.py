from music21 import *
import glob
import ntpath
import numpy as np
from os import listdir
from keras.models import model_from_json
from keras_preprocessing import sequence
import csv

song_path = 'showdata/MIDI/Ode_to_Joy_Easy_variation.mxl'
melody = converter.parse(song_path, format='musicxml')
melody.show()
