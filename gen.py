from pre import *

start = numpy.random.randint(0, len(network_input) - 1)
print(start)
batch_size = 1
test_model = LSTMGen(sequence_length, 512, n_vocab, batch_size)
test_model_state_dict = torch.load("model.pth")
test_model.load_state_dict(test_model_state_dict)
test_model.to(device)


int_to_note = dict((number, note) for number, note in enumerate(pitch_names))
pattern = network_input[start: start + batch_size]

prediction_output = []

with torch.no_grad():
    for i, note_index in enumerate(range(500)):
        prediction_input = np.reshape(pattern, (1, batch_size, sequence_length))
        prediction_input = torch.Tensor(prediction_input).to(device)
        prediction = test_model(prediction_input)
        index = torch.argmax(prediction[0][0]).item()
        result = int_to_note[index]
        if result == 'start' or result == 'end':
            continue
        prediction_output.append(result)
        pattern[:-1] = pattern[1:]
        pattern[-1] = index


offset = 0
output_notes = []

for pattern in prediction_output:

    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        midi_notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()
            midi_notes.append(new_note)
        new_chord = chord.Chord(midi_notes)
        new_chord.offset = offset
        output_notes.append(new_chord)

    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)

    offset += 0.5

midi_stream = stream.Stream(output_notes)

midi_stream.write('midi', fp='res.mid')