from music21 import *
import glob
import ntpath
import numpy as np
from os import listdir
from keras.models import model_from_json
from keras_preprocessing import sequence
import csv
#此函数为借鉴模型示例
def load_model():
    # load model file
    model_dir = 'model_json/'
    model_files = listdir(model_dir)
    for i, file in enumerate(model_files):
        print(str(i) + " : " + file)
    file_number_model = int(input('Choose the model:'))
    model_file = model_files[file_number_model]
    model_path = '%s%s' % (model_dir, model_file)

    # load weights file
    weights_dir = 'model_weights/'
    weights_files = listdir(weights_dir)
    for i, file in enumerate(weights_files):
        print(str(i) + " : " + file)
    file_number_weights = int(input('Choose the weights:'))
    weights_file = weights_files[file_number_weights]
    weights_path = '%s%s' % (weights_dir, weights_file)

    # load the model
    model = model_from_json(open(model_path).read())
    model.load_weights(weights_path)

    return model


def predict():


    chord_list = {'C:maj':chord.Chord(["C4", "E4", "G4"]), 'C:min':chord.Chord(["C4", "E4b", "G4"]),
                        'C#:maj':chord.Chord(["C4#", "E4#", "G4#"]), 'C#:min':chord.Chord(["C4#", "E4", "G4#"]),
                        'D:maj': chord.Chord(["D4", "F4#", "A4"]), 'D:min':chord.Chord(["D4", "F4", "A4"]),
                        'D#:maj': chord.Chord(["D4#", "G4", "A4#"]), 'D#:min': chord.Chord(["D4#", "F4#", "A4#"]),
                        'E:maj':chord.Chord(["E4", "G4#", "B4"]), 'E:min':chord.Chord(["E4", "G4", "B4"]),
                        'F:maj':chord.Chord(["F3", "A3", "C4"]), 'F:min':chord.Chord(["F3", "A3b", "C4"]),
                        'F#:maj': chord.Chord(["F3#", "A3#", "C4#"]), 'F#:min': chord.Chord(["F3#", "A3", "C4#"]),
                        'G:maj': chord.Chord(["G3", "B3", "D4"]), 'G:min':chord.Chord(["G3", "B3b", "D4"]),
                        'G#:maj':chord.Chord(["G3#", "B3#", "D4#"]), 'G#:min':chord.Chord(["G3#", "B3", "D4#"]),
                        'A:maj':chord.Chord(["A3", "C4#", "E4"]), 'A:min': chord.Chord(["A3", "C4", "E4"]),
                        'A#:maj': chord.Chord(["A3#", "D4", "E4#"]), 'A#:min':chord.Chord(["A3#", "C4#", "E4#"]),
                        'B:maj': chord.Chord(["B3", "D4#", "F4#"]), 'B:min':chord.Chord(["B3", "D4", "F4#"])}
    node_list = ['C:maj', 'C:min',
                        'C#:maj', 'C#:min',
                        'D:maj', 'D:min',
                        'D#:maj', 'D#:min',
                        'E:maj', 'E:min',
                        'F:maj', 'F:min',
                        'F#:maj', 'F#:min',
                        'G:maj', 'G:min',
                        'G#:maj', 'G#:min',
                        'A:maj', 'A:min',
                        'A#:maj', 'A#:min',
                        'B:maj', 'B:min']
    G_maj_chord={
        'I':chord.Chord(["G3", "B3", "D4"]),'II':chord.Chord(["A3", "C4", "E4"]),
        'III':chord.Chord(["B3", "D4", "F#4"]),'IV':chord.Chord(["C4", "E4", "G4"]),
        'V': chord.Chord(["D4", "F4#", "A4"]),'VI': chord.Chord(["E4", "G4#", "B4"]),
        'VII': chord.Chord(["F3", "A3#", "C4"])
    }
    G_maj_chord_str = {
        'I':(["G3", "B3", "D4"]), 'II':(["A3", "C4", "E4"]),
        'III':(["B3", "D4", "F#4"]), 'IV':(["C4", "E4", "G4"]),
        'V':(["D4", "F4#", "A4"]), 'VI':(["E4", "G4#", "B4"]),
        'VII':(["F3", "A3#", "C4"])
    }
    maj_list=['I','V','IV']
    min_list = ['VII', 'II','III','VI']


    model = load_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    file_path = 'dataset/show_npy/result.npy'
    song_path='showdata/MIDI/Ode_to_Joy_Easy_variation.mxl'



    npy_files = glob.glob(file_path)
    for song in npy_files:
        note_sequence = sequence.pad_sequences(np.load(song), maxlen=32)

        # predict
        prediction_list = []
        net_output = model.predict(note_sequence)
        i=0
        for chord_index in net_output.argmax(axis=1):
            prediction_list.append(node_list[chord_index])
        print(ntpath.basename(song), prediction_list)
        chord_list = []
        for each_chord in prediction_list:
            if i == 0:
                if (prediction_list[i]) == 'G:maj':
                    chord0 = G_maj_chord
                    chord_list.append('I')
            else:
                print(i)
                print(chord_list)
                if chord_list[i-1] in ['I']:
                    may1=['IV','II','VI']
                elif chord_list[i-1] in ['IV','II','VI']:
                    may1 = ['V', 'III', 'VII']
                elif chord_list[i-1] in ['III','V','VII'] :
                    may1 = ['I']
                print(may1)
                may0=[]
                for a in G_maj_chord_str:
                    for b in G_maj_chord_str[a]:
                        if prediction_list[i].split(':')[0] in b:
                            may0.append(a)
                            break
                print(may0)
                if prediction_list[i].split(':')[0]=='min':
                    may2=min_list
                else:
                    may2=maj_list
                print(may2)
                if len(may0)>1:
                    finalmay = [val for val in may0 if val in may2]

                if len(finalmay) > 1:
                    finaltemp = [val for val in may1 if val in finalmay]
                    if len(finaltemp)!=0:
                        finalmay=finaltemp
                print(finalmay)
                if chord_list[i - 1]==finalmay[0] and len(finalmay)>1:
                    chord_list.append(finalmay[1])
                else :
                    chord_list.append(finalmay[0])
            i+=1
        print(ntpath.basename(song), chord_list)
    i = 0
    j = 0
    s = stream.Score(id='mainScore')
    p0 = stream.Part(id='part0')
    p1 = stream.Part(id='part1')
    stream0 = stream.Stream()

    melody=converter.parse(song_path)
    p0=melody.parts[0]

    j = int(0)
    stream1 = stream.Stream()
    stream0.recurse()
    for index in chord_list:
         #stream1.measure(j).append(chord_list[index])
         m = stream.Measure(number=j)
         final_chord=G_maj_chord[index]
         final_chord.duration.quarterLength =4
         m.append(final_chord)

         j+=1
         stream1.append(m)
         #stream0.measure(j).append(m)
    # part.append(stream1)
    p1=stream1
    #
    # part.show()
    s.insert(0, p0)
    s.insert(0, p1)
    s.show()




if __name__ == '__main__':
    predict()
