from sequant_funcs import generate_rdkit_descriptors, SeQuant_encoding, generate_latent_representations
import pandas as pd
import numpy as np

max_peptide_length = 96
polymer_type = 'protein'


def minmax_normalization(values):
    min_x, max_x = min(values), max(values)
    delta = max_x - min_x
    return [(x-min_x)/delta * 2 - 1 for x in values]


def get_dataset(filename,
                sense_column='Sense',
                antisense_column='AntiSense',
                concentration_column='Concentration, nM',
                efficacy_column='Efficacy, %'):

    df = pd.read_csv(filename)

    senses = df[sense_column].to_list()
    antisenses = df[antisense_column].to_list()
    concs = df[concentration_column].to_list()
    effs = df[efficacy_column].to_list()

    descriptors_set = generate_rdkit_descriptors()

    norm_effs = minmax_normalization(effs)
    norm_concs = minmax_normalization(concs)

    encoded_senses = SeQuant_encoding(sequences_list=senses,
                                      polymer_type=polymer_type,
                                      descriptors=descriptors_set,
                                      num=max_peptide_length)

    x_senses = generate_latent_representations(sequences_list=encoded_senses,
                                               sequant_encoded_sequences=encoded_senses,
                                               polymer_type=polymer_type,
                                               path_to_model_folder='Models/proteins')

    encoded_antisenses = SeQuant_encoding(sequences_list=antisenses,
                                          polymer_type=polymer_type,
                                          descriptors=descriptors_set,
                                          num=max_peptide_length)

    x_antisenses = generate_latent_representations(sequences_list=encoded_antisenses,
                                                   sequant_encoded_sequences=encoded_antisenses,
                                                   polymer_type=polymer_type,
                                                   path_to_model_folder='Models/proteins')

    x = list()
    for i in range(len(x_senses)):
        m = np.hstack([x_senses[i], x_antisenses[i]])
        m = np.append(m, norm_concs[i])
        x.append(m)
    x = np.vstack(x)
    y = np.array(norm_effs)

    print('\nX.shape', x.shape)
    print('y.shape', y.shape, end='\n\n')

    return x, y
