from music21 import *
import csv
environment.set('musicxmlPath', 'C:/Program Files/MuseScore 3/bin/MuseScore3.exe')


i=0
j=0
stream0=stream.Stream()
with open('dataset/csv_test/502 Blues.csv',"rt", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    #m = stream.Measure(number=0)

    for row in reader:
        if i!=row['measure']:
            i = row['measure']
            m = stream.Measure(number=j)
            j+=1
            if m.number!=0:
                stream0.append(m)
            print(row['measure'])
        print((row['note_root'].rstrip('0')+row['note_octave']))
        if row['note_root']!='rest':
            f = note.Note(row['note_root'].rstrip('0')+row['note_octave'])
        else:
            f = note.Rest()
        f.duration.quarterLength =float(row['note_duration'])/4
        m.append(f)
    #stream0.show()
    #stream1=stream0.makeMeasures()
    fp = stream0.write('musicxml', fp='showdata/MIDI/test.xml')
    stream0.show('text')
