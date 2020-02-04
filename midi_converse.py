import csv
from music21 import converter, instrument, note, chord, stream

f = open('showdata/csv/result.csv', 'w', encoding='utf-8')

csv_writer = csv.writer(f, lineterminator='\n')

csv_writer.writerow(["measure", "chord", "node"])

midi = converter.parse("showdata/MIDI/Ode_to_Joy_Easy_variation.mxl")

notes_to_parse = midi.flat.notes
for element in notes_to_parse:
    if element.isNote:
        print(element.fullName)
        # print(element.beat)
        # print(element.measureNumber)如果对象在小节中，则返回包含此对象的的小节编号。如果对象不在度量范围内，则返回None
        # print(element.name)
        # print(element.octave)
        # print(element.quarterLength)
        if 'D-' in element.name:
            csv_writer.writerow([element.measureNumber, 'C:maj', 'C#'])
        elif 'E-' in element.name:
            csv_writer.writerow([element.measureNumber, 'C:maj', 'D#'])
        elif 'G-' in element.name:
            csv_writer.writerow([element.measureNumber, 'C:maj', 'F#'])
        elif 'A-' in element.name:
            csv_writer.writerow([element.measureNumber, 'C:maj', 'G#'])
        elif 'B-' in element.name:
            csv_writer.writerow([element.measureNumber, 'C:maj', 'A#'])
        else:
            csv_writer.writerow([element.measureNumber, 'C:maj', element.name])

f.close()
