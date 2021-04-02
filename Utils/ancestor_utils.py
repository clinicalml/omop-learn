import psycopg2, pandas as pd, numpy as np, pickle, scipy, time, sys

'''
The main function to call is get_ancestor_matrix
'''

def get_ancestors(feature_codes, features_are_drugs, hierarchical):
    '''
    feature_codes is a list of strings.
    features_are_drugs is a boolean. True if features are drugs. False otherwise.
    hierarchical is a boolean. True if recursively get features
    returns 1) a dictionary that maps each string in feature_codes to its immediate ancestors' codes if present
        and each ancestor to its immediate ancestor until top of hierarchy is reached if hierarchical
        2) inverse mapping of the dictionary above: ancestor to all the features that map to it
    '''
    conn = psycopg2.connect("dbname=omop_v4 user=cji password=* host=/var/run/postgresql")
    sql_command_before_feat_code = 'select distinct concept_id_2 from cdm.concept_relationship ' \
        + 'where concept_id_1 = '
    if features_are_drugs:
        sql_command_after_feat_code = ' and relationship_id = \'RxNorm is a\';'
    else:
        sql_command_after_feat_code = ' and relationship_id = \'Is a\';'
    feature_anc_dict = dict()
    anc_to_feature_dict = dict()
    feats_to_check = set(feature_codes)
    while len(feats_to_check) > 0:
        feat_code = feats_to_check.pop()
        sql_command = sql_command_before_feat_code + str(feat_code) + sql_command_after_feat_code
        ancestor_codes = pd.read_sql(sql_command, conn)
        ancestors = ancestor_codes['concept_id_2'].values.tolist()
        if len(ancestors) > 0:
            feature_anc_dict[feat_code] = ancestors
            for anc in ancestors:
                if anc in anc_to_feature_dict:
                    anc_to_feature_dict[anc].append(feat_code)
                else:
                    anc_to_feature_dict[anc] = [feat_code]
            if hierarchical:
                for anc in ancestors:
                    if anc not in feature_anc_dict:
                        feats_to_check.add(anc)
    return feature_anc_dict, anc_to_feature_dict
    
def get_ancestor_feature_matrix(feature_matrix, feature_codes, anc_to_feature_dict):
    '''
    feature_matrix contains the counts of each feature (number of patients by length of feature_names)
    feature_codes is a list of strings that are the columns in feature_matrix
    anc_to_feature_dict is a dictionary mapping each ancestor to its immediate children
    returns 1) a matrix that is number of patients by length of ancestor_names
               filled with sum of entries in feature_matrix that map to the ancestor in that column
            2) a list of ancestor_codes corresponding to the columns in the matrix above
    '''
    assert feature_matrix.shape[1] == len(feature_codes)
    anc_codes = list(anc_to_feature_dict.keys())
    anc_matrix = np.zeros((feature_matrix.shape[0], len(anc_codes)))
    for i in range(len(anc_codes)):
        feats_counted = set()
        feats_to_count = set(anc_to_feature_dict[anc_codes[i]])
        while len(feats_to_count) != 0:
            feat = feats_to_count.pop()
            if feat in feature_codes:
                feat_idx = feature_codes.index(feat)
                anc_matrix[:,i:i+1] += feature_matrix[:,feat_idx:feat_idx+1]
            feats_counted.add(feat)
            if feat in anc_to_feature_dict:
                for child in anc_to_feature_dict[feat]:
                    if child not in feats_counted:
                        feats_to_count.add(child)
    return anc_matrix, anc_codes

def convert_code_dict_to_names(code_dict, code_to_name_mapping):
    '''
    code_dict is a dictionary that maps each code to a list of codes
    code_to_name_mapping is a dictionary maps each code to a string name
    '''
    name_dict = dict()
    for key_code in code_dict:
        key_name = code_to_name_mapping[key_code]
        value_names = []
        for value_code in code_dict[key_code]:
            value_names.append(code_to_name_mapping[value_code])
        name_dict[key_name] = value_names
    return name_dict

def get_names(anc_codes, feat_to_anc_code_dict, anc_to_feat_code_dict, feat_codes, feat_names):
    '''
    anc_codes is a list of ancestor concept IDs that corresponds to the keys in anc_to_feat_code_dict
    feat_to_anc_code_dict is a dictionary mapping each feature code to a list of ancestor codes
    anc_to_feat_code_dict is a dictionary mapping each ancestor code to a list of feature codes, complement of above
    feat_codes is a list of feature concept IDs 
    feat_names is a list of feature names that corresponds to feat_codes
    returns 1) anc_codes translated into names
            2) feat_to_anc_code_dict translated into names
            3) anc_to_feat_code_dict translated into names
    '''
    assert len(feat_codes) == len(feat_names)
    feat_anc_code_to_name_mapping = dict()
    for i in range(len(feat_codes)):
        feat_anc_code_to_name_mapping[feat_codes[i]] = feat_names[i]
    conn = psycopg2.connect("dbname=omop_v4 user=cji password=* host=/var/run/postgresql")
    sql_command_before_code = 'select concept_name from cdm.concept where concept_id='
    anc_names = []
    for code in anc_codes:
        sql_command = sql_command_before_code + str(code) + ';'
        name = pd.read_sql(sql_command, conn).values[0][0]
        anc_names.append(name)
        feat_anc_code_to_name_mapping[code] = name
    feat_to_anc_name_dict = convert_code_dict_to_names(feat_to_anc_code_dict, feat_anc_code_to_name_mapping)
    anc_to_feat_name_dict = convert_code_dict_to_names(anc_to_feat_code_dict, feat_anc_code_to_name_mapping)
    return anc_names, feat_to_anc_name_dict, anc_to_feat_name_dict

def get_ancestor_matrix_and_mapping(feature_matrix, feature_codes, feature_names, features_are_drugs):
    '''
    feature_matrix contains the counts of each feature (number of patients by length of feature_names)
    feature_codes is a list of concept IDs (strings) that are the columns in feature_matrix
    feature_names is a list of concept names corresponding to the IDs
    features_are_drugs is a boolean for whether the features are drugs
    returns 1) a matrix that is number of patients by length of ancestor_names
               filled with sum of entries in feature_matrix that map to the ancestor in that column
               Example: A -> B -> D means counts of A will contribute to counts of both B and D
            2) a list of ancestor codes corresponding to the columns in the matrix above
            3) a list of ancestor names corresponding to codes above
            4) a dictionary mapping each feature code to its immediate ancestors' codes 
               and each ancestor's code to its ancestors' codes up the hierarchy if present
            5) a dictionary that does the inverse mapping to above
            6) a dictionary mapping names corresponding to 4
            7) a dictionary mapping names corresponding to 5
    '''
    assert feature_matrix.shape[1] == len(feature_codes)
    assert len(feature_codes) == len(feature_names)
    start_time = time.time()
    feat_to_anc_code_dict, anc_to_feat_code_dict = get_ancestors(feature_codes, features_are_drugs, True)
    print('get_ancestors took ' + str(time.time() - start_time) + ' seconds')
    start_time = time.time()
    anc_matrix, anc_codes = get_ancestor_feature_matrix(feature_matrix, feature_codes, anc_to_feat_code_dict)
    print('get_ancestor_feature_matrix took ' + str(time.time() - start_time)  + ' seconds')
    start_time = time.time()
    anc_names, feat_to_anc_name_dict, anc_to_feat_name_dict = get_names(anc_codes, feat_to_anc_code_dict, \
                                                                        anc_to_feat_code_dict, feature_codes, feature_names)
    print('get_names took ' + str(time.time() - start_time) + ' seconds')
    assert set(anc_names) == set(anc_to_feat_name_dict.keys())
    return anc_matrix, anc_codes, anc_names, feat_to_anc_code_dict, anc_to_feat_code_dict, feat_to_anc_name_dict, \
           anc_to_feat_name_dict
    
def print_ancestors_of(feat, anc_name_dict, depth, printed_path):
    '''
    feat is a string
    anc_name_dict maps feature names to ancestors
    depth is number of tabs that precede feat
    printed_path is a set of feature names that have already been on the path
    prints this feature starting at indentation depth and all its ancestors that are not in printed_path (to avoid cycles)
    '''
    output_str = ''
    for i in range(depth):
        output_str += '\t'
    output_str += '-> ' + feat + '\n'
    if feat in printed_path:
        return output_str
    printed_path.add(feat)
    if feat in anc_name_dict:
        for anc in anc_name_dict[feat]:
            output_str += print_ancestors_of(anc, anc_name_dict, depth + 1, printed_path)
    return output_str
    
def print_ancestors_of_frequent_features(feature_matrix, feature_names, anc_name_dict, output_file):
    '''
    feature_matrix is # patients x len(feature_names)
    feature_names is a list of strings
    anc_name_dict maps feature names to ancestors
    output_file is filename string
    if feature occurs in at least 10 patients, print feature name, number of patients with it, and its chain of ancestors
    A (10)
        -> B
            -> D
            -> E
        -> C
    Stops printing cycles when they start to repeat
    '''
    feat_counts_arr = np.sum(np.where(feature_matrix > 0, 1, 0), axis=0)
    freq_idxs = np.nonzero(np.where(feat_counts_arr > 10, 1, 0))[0]
    output_str = ''
    for i in freq_idxs:
        output_str += feature_names[i] + ' (' + str(feat_counts_arr[i]) + ')\n'
        if feature_names[i] in anc_name_dict:
            for anc in anc_name_dict[feature_names[i]]:
                output_str += print_ancestors_of(anc, anc_name_dict, 1, set())
    with open(output_file, 'w') as f:
        f.write(output_str)
        
def print_samples(feature_matrix, feature_names, anc_matrix, anc_names, num_samples, output_file):
    '''
    feature_matrix is # patients by features
    feature_names is a list that corresponds to the columns of feature_matrix
    anc_matrix is # patients by ancestors
    anc_names is a list that corresponds to the columns of anc_matrix
    num_samples is # of random samples to print
    output_file is output path
    '''
    assert feature_matrix.shape[0] == anc_matrix.shape[0]
    assert feature_matrix.shape[1] == len(feature_names)
    assert anc_matrix.shape[1] == len(anc_names)
    assert num_samples <= feature_matrix.shape[0]
    all_idxs = np.arange(feature_matrix.shape[0])
    np.random.shuffle(all_idxs)
    sample_idxs = all_idxs[:num_samples]
    output_str = ''
    for sample_idx in sample_idxs:
        output_str += 'Sample ' + str(sample_idx) + ' features\n'
        feat_idxs = np.nonzero(feature_matrix[sample_idx])[1]
        for feat_idx in feat_idxs:
            output_str += feature_names[feat_idx] + ': ' + str(feature_matrix[sample_idx, feat_idx]) + '\n'
        output_str += '\nSample ' + str(sample_idx) + ' ancestors\n'
        anc_idxs = np.nonzero(anc_matrix[sample_idx])[1]
        for anc_idx in anc_idxs:
            output_str += anc_names[anc_idx] + ': ' + str(anc_matrix[sample_idx, anc_idx]) + '\n'
        output_str += '\n'
    with open(output_file, 'w') as f:
        f.write(output_str)
    
def get_ancestor_matrix(n_days, feature_matrix, feature_names, fileheader):
    '''
    n_days is an integer that matches one of the feature windows in the pickle file
    feature_matrix: sparse matrix containing feature counts: # features x # patients
    feature_names: list of strings corresponding to rows in feature_matrix
    fileheader: start of filenames where outputs are written to
    Run the method above for drugs, conditions, and procedures in the past n_days in our first-line diabetes dataset
    Prints the ancestors hierarchically for each feature
    Merges overlapping ancestors among conditions and procedures
    Prints the features and ancestors for some random samples
    Stores a new pickle file with the ancestor matrix, codes, names, mapping from feature to ancestor codes, 
        mapping from ancestor to feature codes, mapping from feature to ancestor names, 
        mapping from ancestor to feature names, and overlapping ancestors among conditions and procedures
    '''
    feature_matrix = feature_matrix.T.todense()
    if fileheader[-1] != '_':
        fileheader += '_'
    # get the features that correspond to the window n_days
    drug_feature_codes = []
    drug_feature_names = []
    drug_feature_idxs = []
    condition_feature_codes = []
    condition_feature_names = []
    condition_feature_idxs = []
    procedure_feature_codes = []
    procedure_feature_names = []
    procedure_feature_idxs = []
    for idx in range(len(feature_names)):
        feat = feature_names[idx]
        feat_split = feat.split(' - ')
        if feat_split[-1] == str(n_days) + ' days':
            if feat_split[1] == 'drug':
                drug_feature_codes.append(feat_split[0])
                drug_feature_names.append(feat_split[2])
                drug_feature_idxs.append(idx)
            elif feat_split[1] == 'condition':
                condition_feature_codes.append(feat_split[0])
                condition_feature_names.append(feat_split[2])
                condition_feature_idxs.append(idx)
            else:
                procedure_feature_codes.append(feat_split[0])
                procedure_feature_names.append(feat_split[2])
                procedure_feature_idxs.append(idx)
    # get the ancestors for drugs, conditions, and procedures separately
    print('processing drug features')
    drug_anc_matrix, drug_anc_codes, drug_anc_names, drug_feat_to_anc_code_dict, drug_anc_to_feat_code_dict, \
        drug_feat_to_anc_name_dict, drug_anc_to_feat_name_dict \
        = get_ancestor_matrix_and_mapping(feature_matrix[:,drug_feature_idxs], drug_feature_codes, drug_feature_names, True)
    print_ancestors_of_frequent_features(feature_matrix[:,drug_feature_idxs], drug_feature_names, drug_feat_to_anc_name_dict, \
                                         fileheader + 'drug_ancestors_' + str(n_days) + 'days.txt')
    print('processing condition features')
    condition_anc_matrix, condition_anc_codes, condition_anc_names, condition_feat_to_anc_code_dict, \
        condition_anc_to_feat_code_dict, condition_feat_to_anc_name_dict, condition_anc_to_feat_name_dict \
        = get_ancestor_matrix_and_mapping(feature_matrix[:,condition_feature_idxs], condition_feature_codes, \
                                          condition_feature_names, False)
    print_ancestors_of_frequent_features(feature_matrix[:,condition_feature_idxs], condition_feature_names, \
                                         condition_feat_to_anc_name_dict, \
                                         fileheader + 'condition_ancestors_' + str(n_days) + 'days.txt')
    print('processing procedure features')
    procedure_anc_matrix, procedure_anc_codes, procedure_anc_names, procedure_feat_to_anc_code_dict, \
        procedure_anc_to_feat_code_dict, procedure_feat_to_anc_name_dict, procedure_anc_to_feat_name_dict \
        = get_ancestor_matrix_and_mapping(feature_matrix[:,procedure_feature_idxs], procedure_feature_codes, \
                                          procedure_feature_names, False)
    print_ancestors_of_frequent_features(feature_matrix[:,procedure_feature_idxs], procedure_feature_names, \
                                         procedure_feat_to_anc_name_dict, \
                                         fileheader + 'procedure_ancestors_' + str(n_days) + 'days.txt')
    assert len(set(drug_anc_names).intersection(set(condition_anc_names))) == 0
    assert len(set(drug_anc_names).intersection(set(procedure_anc_names))) == 0
    # handle ancestors shared between procedures and conditions
    overlapping_names = set(condition_anc_names).intersection(set(procedure_anc_names))
    for overlapping_name in overlapping_names:
        proc_idx = procedure_anc_names.index(overlapping_name)
        cond_idx = condition_anc_names.index(overlapping_name)
        condition_anc_matrix[:,cond_idx] += procedure_anc_matrix[:,proc_idx]
        condition_anc_to_feat_code_dict[condition_anc_codes[cond_idx]] \
            += procedure_anc_to_feat_code_dict[procedure_anc_codes[proc_idx]]
        condition_anc_to_feat_name_dict[overlapping_name] \
            += procedure_anc_to_feat_name_dict[overlapping_name]
        del procedure_anc_to_feat_code_dict[procedure_anc_codes[proc_idx]]
        del procedure_anc_to_feat_name_dict[overlapping_name]
        del procedure_anc_codes[proc_idx]
        del procedure_anc_names[proc_idx]
        if proc_idx < procedure_anc_matrix.shape[1] - 1:
            procedure_anc_matrix = np.hstack((procedure_anc_matrix[:,:proc_idx], procedure_anc_matrix[:,proc_idx+1:]))
        else:
            procedure_anc_matrix = procedure_anc_matrix[:,:proc_idx]
    assert len(set(condition_anc_names).intersection(set(procedure_anc_names))) == 0
    # merge the 3 into a single ancestor matrix/mapping
    anc_matrix = scipy.sparse.csr_matrix(np.hstack((drug_anc_matrix, condition_anc_matrix, procedure_anc_matrix)), dtype=int)
    anc_codes = drug_anc_codes + condition_anc_codes + procedure_anc_codes
    anc_names = drug_anc_names + condition_anc_names + procedure_anc_names
    feat_to_anc_code_dict = drug_feat_to_anc_code_dict
    feat_to_anc_code_dict.update(condition_feat_to_anc_code_dict)
    feat_to_anc_code_dict.update(procedure_feat_to_anc_code_dict)
    anc_to_feat_code_dict = drug_anc_to_feat_code_dict
    anc_to_feat_code_dict.update(condition_anc_to_feat_code_dict)
    anc_to_feat_code_dict.update(procedure_anc_to_feat_code_dict)
    feat_to_anc_name_dict = drug_feat_to_anc_name_dict
    feat_to_anc_name_dict.update(condition_feat_to_anc_name_dict)
    feat_to_anc_name_dict.update(procedure_feat_to_anc_name_dict)
    anc_to_feat_name_dict = drug_anc_to_feat_name_dict
    anc_to_feat_name_dict.update(condition_anc_to_feat_name_dict)
    anc_to_feat_name_dict.update(procedure_anc_to_feat_name_dict)
    with open(fileheader + 'ancestor_' + str(n_days) + 'days.pkl', 'wb') as f:
        pickle.dump((anc_matrix, anc_codes, anc_names, feat_to_anc_code_dict, anc_to_feat_code_dict, feat_to_anc_name_dict, \
                     anc_to_feat_name_dict, overlapping_names), f)
    print_samples(feature_matrix[:,drug_feature_idxs + condition_feature_idxs + procedure_feature_idxs], \
                  drug_feature_names + condition_feature_names + procedure_feature_names, anc_matrix, anc_names, 10, \
                  fileheader + 'ancestor_' + str(n_days) + 'days_10samples.txt')