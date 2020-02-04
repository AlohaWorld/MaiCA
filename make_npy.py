'''将训练集测试集转换为npy文件，效率更高'''

import glob
import csv
import os
import ntpath
import numpy as np

note_dictionary = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
chord_dictionary = ['C:maj', 'C:min',
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


# one_hot编码部分为参考
def encoding(length, one_index):
    vectors = [0] * length
    vectors[one_index] = 1
    return vectors


# 文件处理有部分参考，已经无法细分
def main():
    # np.set_printoptions(threshold=np.inf)

    print("1. Train set\n2. Test set\n2. Show set")
    _input = input('选择文件:')
    if _input == '1':
        file_path = 'dataset/new_train/*.csv'
    elif _input == '2':
        file_path = 'dataset/new_test/*.csv'
    elif _input == '3':
        file_path = 'showdata/csv/*.csv'
    else:
        print("input error")
        return None

    csv_files = glob.glob(file_path)
    note_dict_len = len(note_dictionary)
    chord_dict_len = len(chord_dictionary)

    result_input_matrix = []
    result_target_matrix = []

    for csv_path in csv_files:
        csv_ins = open(csv_path, 'r', encoding='utf-8')
        next(csv_ins)
        reader = csv.reader(csv_ins)

        note_sequence = []
        song_sequence = []
        pre_measure = None
        for line in reader:
            measure = int(line[0])
            chord = line[1]
            note = line[2]

            chord_index = chord_dictionary.index(chord)
            note_index = note_dictionary.index(note)

            one_hot_note_vec = encoding(note_dict_len, note_index)
            one_hot_chord_vec = encoding(chord_dict_len, chord_index)

            # pre_measure = None

            if pre_measure is None:
                note_sequence.append(one_hot_note_vec)
                result_target_matrix.append(one_hot_chord_vec)

            elif pre_measure == measure:
                note_sequence.append(one_hot_note_vec)

            else:
                song_sequence.append(note_sequence)
                result_input_matrix.append(note_sequence)
                note_sequence = [one_hot_note_vec]
                result_target_matrix.append(one_hot_chord_vec)
            pre_measure = measure
        result_input_matrix.append(note_sequence)

        if _input == '2':

            file_path = "dataset/test_npy"
            if not os.path.isdir(file_path):
                os.mkdir(file_path)
            np.save('%s/%s.npy' % (file_path, ntpath.basename(csv_path).split('.')[0]), np.array(song_sequence))
        if _input == '3':

            file_path = "dataset/show_npy"
            if not os.path.isdir(file_path):
                os.mkdir(file_path)
            np.save('%s/%s.npy' % (file_path, ntpath.basename(csv_path).split('.')[0]), np.array(song_sequence))

    if _input == '1':
        np.save('dataset/input_vector.npy', np.array(result_input_matrix))
        np.save('dataset/target_vector.npy', np.array(result_target_matrix))
    elif _input == '2':
        np.save('dataset/test_vector.npy', np.array(result_input_matrix))
    elif _input == '3':
        np.save('dataset/show_vector.npy', np.array(result_input_matrix))


if __name__ == '__main__':
    main()
