def is_ctro_data_string(string):
    return string.startswith('<http://ctro/data#')


def is_rdf_type_predicate(predicate):
    return predicate.endswith('rdf-syntax-ns#type>')


def is_rdf_label_predicate(predicate):
    return predicate.endswith('rdf-schema#label>')


def extract_rdf_identifier(rdf_string):
    # check if input string contains identifier, i.e., check if input string is ctro data string
    if not is_ctro_data_string(rdf_string):
        raise Exception('ctro data string expected')

    # remove angular brackets (first and last character of input string)
    rdf_string = rdf_string[1:-1]

    # get index of number sign
    number_sign_index = rdf_string.rindex('#')

    # extract and reutn identifier
    return rdf_string[number_sign_index + 1:]
