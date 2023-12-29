from sequant_funcs import *
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

max_peptide_length = 96
polymer_type = 'protein'


def normalize(values, feature_range=(-1, 1)):
    sc = MinMaxScaler(feature_range=feature_range)
    values = np.array([values])
    values = values.transpose()
    values = [i[0] for i in sc.fit_transform(values)]

    values = tf.constant(values, dtype=tf.float32)
    values = tf.expand_dims(values, axis=1)

    return values


def get_dataset(filename,
                sense_column='Sense',
                antisense_column='Antisense',
                concentration_column='Concentration, nM',
                efficacy_column='Efficacy, %'):

    df = pd.read_csv(filename)
    # df = df.loc[df[concentration_column] < 1000]

    sequences = [(line[sense_column], line[antisense_column]) for _, line in df.iterrows()]
    concentration_pack = df[concentration_column].to_list()
    efficacy_pack = df[efficacy_column].to_list()
    descriptors_set = generate_rdkit_descriptors()

    sense_pack, antisense_pack = list(), list()

    for sense, antisense in sequences:
        sense_matrix = seq_to_matrix_(sequence=sense,
                                      polymer_type=polymer_type,
                                      descriptors=descriptors_set,
                                      num=max_peptide_length)

        antisense_matrix = seq_to_matrix_(sequence=antisense,
                                          polymer_type=polymer_type,
                                          descriptors=descriptors_set,
                                          num=max_peptide_length)

        sense_matrix = tf.expand_dims(sense_matrix, axis=0)
        antisense_matrix = tf.expand_dims(antisense_matrix, axis=0)

        sense_pack.append(sense_matrix)
        antisense_pack.append(antisense_matrix)

    sense_pack = tf.concat(sense_pack, axis=0)
    antisense_pack = tf.concat(antisense_pack, axis=0)

    compressed_sense = generate_latent_representations(sequences_list=df['Sensesequence'].to_list(),
                                                       sequant_encoded_sequences=sense_pack,
                                                       polymer_type=polymer_type,
                                                       add_peptide_descriptors=False,
                                                       path_to_model_folder='Models/proteins')

    compressed_antisense = generate_latent_representations(sequences_list=df['Antisensesequence'].to_list(),
                                                           sequant_encoded_sequences=antisense_pack,
                                                           polymer_type=polymer_type,
                                                           add_peptide_descriptors=False,
                                                           path_to_model_folder='Models/proteins')

    merged_compressed_sequences = tf.concat([compressed_sense, compressed_antisense], axis=1)

    concentrations = normalize(concentration_pack)
    efficacy_pack = normalize(efficacy_pack)

    x = tf.concat([merged_compressed_sequences, concentrations], axis=1)
    y = efficacy_pack

    print()
    print('X shape:', x.shape)
    print('y shape:', y.shape, end='\n\n')

    x = [np.array([i]) for i in x]
    y = [np.array(i) for i in y]

    print(y)

    return x, y
