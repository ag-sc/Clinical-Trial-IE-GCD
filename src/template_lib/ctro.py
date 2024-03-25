import csv
import numpy as np
import os
import random


# os.chdir(os.path.dirname(__file__))


def extract_identifier(string):
    index = string.index('#')
    return string[index + 1:-1]


def check_pairwise_disjoint(set_list):
    for i in range(len(set_list) - 1):
        for j in range(i + 1, len(set_list)):
            set1 = set_list[i]
            set2 = set_list[j]

            if len(set1.intersection(set2)) > 0:
                return False

    return True


class CtroOntology:
    group_names = ['Publication', 'Population', 'ClinicalTrial', 'EvidenceQuality',
                   'DiffBetweenGroups', 'Arm', 'Intervention', 'Medication', 'Outcome', 'Endpoint']

    used_group_names = ['Publication', 'Population', 'ClinicalTrial',
                        'DiffBetweenGroups', 'Arm', 'Intervention', 'Medication', 'Outcome', 'Endpoint']

    single_group_names = ['Publication', 'Population', 'ClinicalTrial']
    multi_group_names = ['DiffBetweenGroups', 'Arm', 'Intervention', 'Medication', 'Outcome', 'Endpoint']

    high_level_slot_names = ['describes', 'hasArm', 'hasPopulation', 'hasDiffBetweenGroups', 'hasEvidQualityIndicator',
                             'hasOutcome1', 'hasOutcome2', 'hasOutcome', 'hasAdverseEffect', 'hasIntervention',
                             'hasMedication', 'hasEndpoint']

    def __init__(self, folder):
        # contains the slot names for each group name
        self.group_slots = {}
        for group_name in self.group_names:
            self.group_slots[group_name] = set()

        # dict containing the corresponding group for each slot
        self.group_of_slot = {}

        # contains the class of each slot
        self.slot_type = {}

        # set of classes which are individuals
        self.individuals = set()

        # most general superclasses
        self.most_general_superclass = {}

        # set of all subclasses for a given class; class_name -> set(classes)
        # including indirect subclasses and query class
        self.all_subclasses = {}

        # ORDERING IS IMPORTATNT
        self.used_most_general_classes = []

        # used slots
        self.used_slots = set()

        # group name -> slot name -> slot index
        self.slot_indices = {}

        # group_name -> slot index -> slot name
        self.slot_indices_reverse = {}

        # indices most used entity classes; class- > index / 
        self.used_most_general_classes_indices = {}

        # index -> class name
        self.used_most_general_classes_indices_reverse = {}

        # group name -> class name -> active slots mask
        self.active_slot_masks = {}

        # import group definitions ###################################
        f = open(os.path.join(folder, 'resources/properties.csv'))
        f.readline()
        for line in f:
            line = line.strip()
            cols = line.split('\t')

            self.group_slots[cols[0]].add(cols[1])
            self.group_of_slot[cols[1]] = cols[0]
            self.slot_type[cols[1]] = cols[2]

        # import most general superclasses ############################
        f = open(os.path.join(folder, 'resources/most_general_superclasses.csv'))
        for line in f:
            line = line.strip()
            cols = line.split('\t')
            self.most_general_superclass[cols[0]] = cols[1]

        values = list(self.most_general_superclass.values())
        for value in values:
            self.most_general_superclass[value] = value
        for c in self.used_most_general_classes:
            self.most_general_superclass[c] = c
        for slot_name in self.slot_type:
            slot_type = self.slot_type[slot_name]
            if slot_type not in self.most_general_superclass:
                self.most_general_superclass[slot_type] = slot_type

        # used most general classes ############################ 
        self.used_most_general_classes = set(self.most_general_superclass.values())

        # all subclasses #######################################
        f = open(os.path.join(folder, 'resources/subclasses.csv'))
        for line in f:
            line = line.strip()
            cols = line.split('\t')

            superclass = cols[0]
            subclass = cols[1]

            if superclass not in self.all_subclasses:
                self.all_subclasses[superclass] = set()

            self.all_subclasses[superclass].add(subclass)

        # indirect subclasses
        for superclass in self.all_subclasses:
            subclasses_to_check = self.all_subclasses[superclass]
            new_subclasses = set()

            while True:
                for subclass in subclasses_to_check:
                    if subclass in self.all_subclasses:
                        for c in self.all_subclasses[subclass]:
                            if c not in self.all_subclasses[superclass]:
                                new_subclasses.add(c)

                if len(new_subclasses) == 0:
                    break

                self.all_subclasses[superclass].update(new_subclasses)
                subclasses_to_check = new_subclasses
                new_subclasses = set()

        # estimate which slots can be filled by most general superclasses ###
        for group_name in self.group_names:
            for slot_name in self.group_slots[group_name]:
                slot_type = self.slot_type[slot_name]

                superclass = None
                if slot_type in self.most_general_superclass:
                    superclass = self.most_general_superclass[slot_type]

                self.used_slots.add(slot_name)

        # create slot indices ##########################################
        for group_name in self.group_names:
            self.slot_indices[group_name] = dict()
            self.slot_indices_reverse[group_name] = dict()

            for slot_name in self.group_slots[group_name]:
                if slot_name in self.used_slots:
                    self.slot_indices[group_name][slot_name] = len(self.slot_indices[group_name])
                    self.slot_indices_reverse[group_name][len(self.slot_indices_reverse[group_name])] = slot_name

        # class indices ##############################################
        for index, class_name in enumerate(self.used_most_general_classes):
            self.used_most_general_classes_indices[class_name] = index
            self.used_most_general_classes_indices_reverse[index] = class_name

        # active slots masks ###################################################
        for group_name in self.group_names:
            self.active_slot_masks[group_name] = {}

            for class_name in self.used_most_general_classes:
                slot_mask = np.zeros((len(self.slot_indices[group_name])))

                for slot_name in self.slot_indices[group_name]:
                    if self.most_general_superclass[self.slot_type[slot_name]] == class_name:
                        slot_mask[self.slot_indices[group_name][slot_name]] = 1

                self.active_slot_masks[group_name][class_name] = slot_mask


class Tokenization:
    def __init__(self):
        # list of (doc_char_onset,sentence_number,token)
        self.tokens = []

    def set_from_file(self, filename):
        self.tokens = []
        f = open(filename)
        f.readline()  # first line is comment
        csv_reader = csv.reader((line.replace('\0', '') for line in f), delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_ALL, skipinitialspace=True)

        for cols in csv_reader:
            doc_char_onset = int(cols[6].strip())
            sentence_number = int(cols[1].strip())
            token = cols[8].strip().lower()
            self.tokens.append((sentence_number, doc_char_onset, token))

    def get_sentence_with_doc_char_onset(self, query_onset):
        found_sentence_number = None

        for (sentence_number, doc_char_onset, token) in self.tokens:
            if doc_char_onset == query_onset:
                found_sentence_number = sentence_number

        if found_sentence_number == None:
            raise Exception('No Sentence with specified onset token found: ' + str(query_onset))

        result = []
        for (sentence_number, doc_char_onset, token) in self.tokens:
            if sentence_number == found_sentence_number:
                result.append((sentence_number, doc_char_onset, token))

        return result


class Annotation:
    def __init__(self):
        self.tokens = []
        self.left_context = []
        self.right_context = []
        self.label = None
        self.most_general_superclass = None
        self.sentence_number = None
        self.referencing_slots = []  # list of pairs (group_id,slot_name)
        self.within_class_index = None  # index among all annotations of most_general_class
        self.global_index = None  # global index among all annotations of an abstract
        self.doc_char_onset = None
        # self.doc_char_offset = None


def do_ranges_overlap(range1, range2):
    start1, end1 = range1
    start2, end2 = range2

    set1 = set(range(start1, end1 + 1))
    set2 = set(range(start2, end2 + 1))

    return len(set1.intersection(set2)) > 0


class AnnotatedAbstract():
    def __init__(self):
        self.annotations = []
        self.class_subdivision_annotations = {}

    def set_from_file(self, filename, tokenization):
        self.annotations = []
        f = open(filename)
        f.readline()  # first line is comment
        csv_reader = csv.reader(f, delimiter=',', quotechar='"')

        # create class count dict for assigning within class indices
        class_counts = {}
        for c in ontology.used_most_general_classes:
            class_counts[c] = 0

        # process annotations
        for cols in csv_reader:
            # check if label is subclass of most general class #####################
            label = cols[1].strip()

            if label not in ontology.most_general_superclass:
                continue

            superclass = ontology.most_general_superclass[label]

            # get tokens of annotation and left and right context ################
            annotation = Annotation()
            doc_onset_annotation = int(cols[2].strip())
            doc_offset_annotation = int(cols[3].strip())
            annotation.doc_char_onset = doc_onset_annotation
            sentence = tokenization.get_sentence_with_doc_char_onset(doc_onset_annotation)
            annotation.sentence_number = sentence[0][0]

            for (sentence_number, doc_onset_token, token) in sentence:
                if doc_onset_token < doc_onset_annotation:
                    annotation.left_context.append(token)
                elif doc_onset_token >= doc_offset_annotation:
                    annotation.right_context.append(token)
                else:
                    annotation.tokens.append(token)

            # add further information #############################################
            annotation.label = label
            annotation.most_general_superclass = superclass

            annotation.within_class_index = class_counts[superclass]
            class_counts[superclass] += 1

            # instances /referencing slots #######################################
            instances = cols[6].strip()
            instances = instances.replace('www.w3.org', '')
            instances = instances[1:-1]
            csv_instances = csv.reader([instances], delimiter='.')
            for instances in csv_instances:
                for instance in instances:
                    csv_triple = csv.reader([instance], delimiter=' ', quotechar='\\')

                    for triple_cols in csv_triple:
                        if len(triple_cols) == 3:
                            if '#' not in triple_cols[0] or '#' not in triple_cols[1]:
                                continue

                            group_id = extract_identifier(triple_cols[0])
                            slot_name = extract_identifier(triple_cols[1])
                            annotation.referencing_slots.append((group_id, slot_name))

            self.annotations.append(annotation)

        # set global index of each annotation
        for i, annotation in enumerate(self.annotations):
            annotation.global_index = i

    # removes annotations which lables are not a subclass of the used most general classes
    # and removes unused referencing slots    
    def shrink(self):
        annotations_new = []

        for annotation in self.annotations:
            if annotation.most_general_superclass not in ontology.used_most_general_classes:
                continue

            referencing_slots_new = []

            # remove references to slots which type is not subclass of used most general classes
            for group_id, slot_name in annotation.referencing_slots:
                if slot_name in ontology.used_slots:
                    referencing_slots_new.append((group_id, slot_name))

            # save updated annotations
            annotation.referencing_slots = referencing_slots_new
            annotations_new.append(annotation)

        self.annotations = annotations_new

    def count_label_occurences(self):
        pass

    def get_annotation_ranges(self, sentence_number, label):
        ranges = []

        for annotation in self.annotations:
            if annotation.sentence_number != sentence_number:
                continue
            if annotation.most_general_superclass != label:
                continue

            annotation_index = annotation.global_index
            start_pos = len(annotation.left_context)
            end_pos = start_pos + len(annotation.tokens) - 1

            ranges.append((annotation_index, start_pos, end_pos))

        return ranges

    def assign_new_global_indices(self):
        for i, annotation in enumerate(self.annotations):
            annotation.global_index = i


class SlotValue:
    def __init__(self):
        self.string = None
        self.annotation = None


class F1Statistics:
    def __init__(self):
        self.num_occurences = 0.0
        self.true_positives = 0.0
        self.false_positives = 0.0

    def get_num_true_positives(self):
        return self.true_positives

    def get_num_false_positives(self):
        return self.false_positives

    def get_num_occurences(self):
        return self.num_occurences

    def get_num_false_negatives(self):
        return self.get_num_occurences() - self.get_num_true_positives()

    def precision(self):
        return self.true_positives / (self.true_positives + self.false_positives + 0.0000000001)

    def recall(self):
        return self.true_positives / (self.num_occurences + 0.0000000001)

    def f1(self):
        precision = self.precision()
        recall = self.recall()
        return (2 * precision * recall) / (precision + recall + 0.0000000001)


class F1StatisticsCollection:
    def __init__(self):
        # slot name -> F1Statistics instance
        self.statistics_dict = {}

        # prediction errors statistics
        self.num_fps_empty_gt_slot = 0.0
        self.num_fns_empty_predicted_slot = 0.0

        self.num_fps_incorrect_class = 0.0

        self.num_errors_incorrect_entity_boundaries = 0.0

    def update(self, group_gt, group_predicted, used_slots=None):
        slot_names = set(group_gt.slots.keys()).union(set(group_predicted.slots.keys()))

        for slot_name in slot_names:
            if used_slots is not None and slot_name not in used_slots:
                continue
            if slot_name not in self.statistics_dict:
                self.statistics_dict[slot_name] = F1Statistics()

            gt = group_gt.slot_values_as_set(slot_name)

            # remover later #############################################
            # if len(gt) == 0:
            # continue
            #############################################################

            predicted = group_predicted.slot_values_as_set(slot_name)

            self.statistics_dict[slot_name].num_occurences += len(gt)
            self.statistics_dict[slot_name].true_positives += len(gt.intersection(predicted))
            self.statistics_dict[slot_name].false_positives += len(predicted.difference(gt))

            # update error statistics
            if len(gt) == 0:
                self.num_fps_empty_gt_slot += len(predicted)
            if len(predicted) == 0:
                self.num_fns_empty_predicted_slot += len(gt)

            if slot_name not in group_gt.slots:
                continue

            for slot_value_gt in group_gt.slots[slot_name]:
                if slot_value_gt.string in predicted:
                    continue

                if slot_value_gt.annotation is not None:
                    partial_match = False

                    for token in slot_value_gt.annotation.tokens:
                        for entity_predicted in predicted:
                            if token in entity_predicted:
                                partial_match = True

                    if partial_match:
                        self.num_errors_incorrect_entity_boundaries += 1

    def update_num_occurences(self, group):
        for slot_name in group.slots:
            if slot_name not in self.statistics_dict:
                self.statistics_dict[slot_name] = F1Statistics()

            self.statistics_dict[slot_name].num_occurences += len(group.slot_values_as_set(slot_name))

    def update_false_positives(self, group):
        for slot_name in group.slots:
            if slot_name not in self.statistics_dict:
                self.statistics_dict[slot_name] = F1Statistics()

            self.statistics_dict[slot_name].false_positives += len(group.slot_values_as_set(slot_name))

    def get_micro_stats(self):
        statistics = F1Statistics()

        for slot_name in self.statistics_dict:
            statistics.num_occurences += self.statistics_dict[slot_name].num_occurences
            statistics.true_positives += self.statistics_dict[slot_name].true_positives
            statistics.false_positives += self.statistics_dict[slot_name].false_positives

        return statistics

    def get_micro_stats_group(self, group_name):
        statistics = F1Statistics()

        for slot_name in ontology.group_slots[group_name]:
            if slot_name not in self.statistics_dict:
                continue

            statistics.num_occurences += self.statistics_dict[slot_name].num_occurences
            statistics.true_positives += self.statistics_dict[slot_name].true_positives
            statistics.false_positives += self.statistics_dict[slot_name].false_positives

        return statistics

    def count_total_flase_positives(self):
        fps = 0.0

        for slot_name in self.statistics_dict:
            fps += self.statistics_dict[slot_name].true_positives

        return fps

    def count_total_false_negatives(self):
        fns = 0.0

        for slot_name in self.statistics_dict:
            fns += self.statistics_dict[slot_name].num_occurences - self.statistics_dict[slot_name].true_positives

        return fns

    def print_statistics(self):
        for group_name in ontology.group_names:
            print(group_name + ' --------------------')
            for slot_name in self.statistics_dict:
                if slot_name in ontology.group_slots[group_name]:
                    print(slot_name + ':')
                    print('(' + str(self.statistics_dict[slot_name].precision()) + ', ' + str(
                        self.statistics_dict[slot_name].recall()) + ', ' + str(
                        self.statistics_dict[slot_name].f1()) + ')')
                    print(' ')

    def print_common_error_statistics(self):
        fps = self.count_total_flase_positives()
        fns = self.count_total_false_negatives()

        print('%fps empty gt slot: ' + str(self.num_fps_empty_gt_slot / fps))
        print('%fns empty predicted slot: ' + str(self.num_fns_empty_predicted_slot / fns))

        print('#fps incorrect entity boundaries: ' + str(self.num_errors_incorrect_entity_boundaries / fps))
        print('#fns incorrect entity boundaries: ' + str(self.num_errors_incorrect_entity_boundaries / fns))


class Group:
    def __init__(self, group_id=None):
        self.slots = {}  # slot name -> list SlotValue
        self.group_id = group_id
        self.group_name = None

    def add_value(self, slot_name, slot_value):
        if slot_name not in self.slots:
            self.slots[slot_name] = []

        self.slots[slot_name].append(slot_value)

    def __str__(self):
        string = self.group_id + ' ------------------\n'

        for slot_name in self.slots:
            string = string + slot_name + ': '

            for slot_value in self.slots[slot_name]:
                string = string + slot_value.string + '(' + slot_value.annotation.most_general_superclass + '); '

            string = string + '\n'

        return string

    # returns global indices of annotations which can be used
    # to compute "keys" of groups
    def get_global_reference_indices(self, annotated_abstract, disease_string):
        if disease_string == 'gl':
            scores_dict = ontology.reference_slots_glaucoma
        elif disease_string == 'dm2':
            scores_dict = ontology.reference_slots_diabetes
        else:
            raise ('Invalid disease String')

        scores_dict = scores_dict[self.group_name]
        indices = []

        for slot_name, score in scores_dict:
            if slot_name in self.slots:
                for slot_value in self.slots[slot_name]:
                    indices.append(slot_value.annotation.global_index)

                return indices

        return []

    def get_key_annotations_fixed(self, annotated_abstract, disease_string, query_slot_name):
        if disease_string == 'gl':
            scores_dict = ontology.reference_slots_glaucoma
        elif disease_string == 'dm2':
            scores_dict = ontology.reference_slots_diabetes
        else:
            raise ('Invalid disease String')

        scores_dict = scores_dict[self.group_name]
        result_annotations = []

        for slot_name, score in scores_dict:
            if slot_name == query_slot_name:
                continue
            if slot_name not in self.slots:
                return []

            for slot_value in self.slots[slot_name]:
                result_annotations.append(slot_value.annotation)

            return result_annotations

    def encode(self, annotated_abstract):
        encoded_group = []
        num_slots = len(ontology.slot_indices[self.group_name])
        slot_indices = ontology.slot_indices[self.group_name]

        for annotation in annotated_abstract.annotations:
            vector = np.zeros((num_slots))

            for group_id, slot_name in annotation.referencing_slots:
                if group_id == self.group_id and slot_name in slot_indices:
                    vector[slot_indices[slot_name]] = 1

            encoded_group.append(vector)

        return encoded_group

    def set_from_encoding(self, encoded_group, annotated_abstract):
        self.slots = {}
        slot_indices_reverse = ontology.slot_indices_reverse[self.group_name]

        for i, annotation in enumerate(annotated_abstract.annotations):
            vector = encoded_group[i]

            for slot_index in range(len(vector)):
                if vector[slot_index] == 1:
                    slot_name = slot_indices_reverse[slot_index]

                    slot_value = SlotValue()
                    slot_value.string = ' '.join(annotation.tokens)
                    slot_value.annotation = annotation

                    self.add_value(slot_name, slot_value)

    def slot_values_as_set(self, slot_name):
        result = set()

        if slot_name not in self.slots:
            return set()

        for slot_value in self.slots[slot_name]:
            result.add(slot_value.string)

        return result

    def to_dict(self):
        dictionary = {}

        for slot_name in self.slots:
            slot_values = self.slot_values_as_set(slot_name)
            dictionary[slot_name] = slot_values

        return dictionary

    def get_entities_as_strings(self):
        entities = set()

        for slot_name in self.slots:
            for slot_value in self.slots[slot_name]:
                if slot_value.string != None:
                    entities.add(slot_value.string)

        return entities

    def get_entities_as_global_annotation_indices(self):
        entities = set()

        for slot_name in self.slots:
            for slot_value in self.slots[slot_name]:
                if slot_value.annotation != None:
                    entities.add(slot_value.annotation.global_index)

        return entities

    def get_slot_values_as_global_annotation_indices(self, slot_name):
        entities = set()

        for slot_value in self.slots[slot_name]:
            if slot_value.annotation != None:
                entities.add(slot_value.annotation.global_index)

        return entities


class GroupCollection:
    def __init__(self):
        # group name -> list of groups
        self.groups = {}

        for group_name in ontology.group_names:
            self.groups[group_name] = []

    def import_group_ids(self, filename):
        f = open(filename)

        for line in f:
            if line.startswith('#'):
                continue

            cols = line.split(' ')
            identifier = extract_identifier(cols[0])
            index = identifier.index('_')

            group_name = identifier[:index]

            # check if group is top-level group
            if group_name not in ontology.group_names:
                continue

            # create group
            group = Group(identifier)
            group.group_name = group_name
            self.groups[group_name].append(group)

    def get_group(self, group_name, group_id):
        if group_name not in self.groups:
            return None

        for group in self.groups[group_name]:
            if group.group_id == group_id:
                return group

        return None

    def get_group_from_id(self, group_id):
        for group_name in self.groups:
            for group in self.groups[group_name]:
                if group.group_id == group_id:
                    return group

        return None

    def get_group_from_id_print(self, group_id):
        for group_name in self.groups:
            for group in self.groups[group_name]:
                print(group.group_id)
                if group.group_id == group_id:
                    return group

        return None

    def get_group_name_ids(self, group_name):
        ids = set()

        if group_name not in self.groups:
            return set()

        for group in self.groups[group_name]:
            ids.add(group.group_id)

        return ids

    # cretaes references to atomic slot fillers, i.e. entities    
    def fill(self, annotated_abstract, remove_empty_groups=True):
        for annotation in annotated_abstract.annotations:
            for group_id, slot_name in annotation.referencing_slots:
                if '_' not in group_id:
                    continue

                index = group_id.index('_')
                group_name = group_id[:index]
                if group_name not in ontology.group_names:
                    continue

                # get coressponding group for reference
                group = self.get_group(group_name, group_id)
                if group == None:
                    print('Slot filling error: Group with ID ' + group_id + ' not found!')
                    continue

                # create SlotValue instance
                slot_value = SlotValue()
                slot_value.string = ' '.join(annotation.tokens)
                slot_value.annotation = annotation

                # store slot value
                group.add_value(slot_name, slot_value)

        # remove groups which have no slot fillers
        for group_name in self.groups:
            new_groups_list = []

            for group in self.groups[group_name]:
                if len(group.slots) == 0 and remove_empty_groups:
                    continue
                else:
                    new_groups_list.append(group)

            self.groups[group_name] = new_groups_list

    def remove_evidence_quality_groups(self):
        del self.groups['EvidenceQuality']

    # creates new instance with same group ids, but not filled    
    def create_structure_copy(self):
        group_collection_copy = GroupCollection()

        for group_name in self.groups:
            for group in self.groups[group_name]:
                group_copy = Group(group.group_id)
                group_copy.group_name = group_name
                group_collection_copy.groups[group_name].append(group_copy)

        return group_collection_copy

    # dict[group_name] -> list encodings
    def encode(self, annotated_abstract):
        encoded_groups = {}

        for group_name in self.groups:
            encoded_groups[group_name] = []

            for group in self.groups[group_name]:
                encoded_group = group.encode(annotated_abstract)
                encoded_groups[group_name].append(encoded_group)

        return encoded_groups

    def set_from_encoded_groups(self, encoded_groups, annotated_abstract):
        self.groups = {}

        for group_name in encoded_groups:
            self.groups[group_name] = []

            for encoded_group in encoded_groups[group_name]:
                group = Group()
                group.group_name = group_name
                group.set_from_encoding(encoded_group, annotated_abstract)
                self.groups[group_name].append(group)

    # removes all empty groups, i.e. groups which have no slot fillers
    def remove_empty_groups(self):
        for group_name in self.groups:
            new_group_list = [group for group in self.groups[group_name] if len(group.slots) > 0]
            self.groups[group_name] = new_group_list

    def import_high_level_slot_values(self, filename):
        f = open(filename)

        for line in f:
            if line.startswith('#'):
                continue

            cols = line.split(' ')
            if '#' not in cols[0] or '#' not in cols[1] or '#' not in cols[2]:
                continue

            subject_id = extract_identifier(cols[0])
            slot_name = extract_identifier(cols[1])
            object_id = extract_identifier(cols[2])

            if slot_name not in ontology.high_level_slot_names:
                continue

            group = self.get_group_from_id(subject_id)
            if group is None:
                print('------------------------')
                print(slot_name)
                print(subject_id)
                self.get_group_from_id_print(subject_id)
                continue

            if slot_name not in group.slots:
                group.slots[slot_name] = []

            slot_value = SlotValue()
            slot_value.string = object_id
            group.slots[slot_name].append(slot_value)

    def align_high_level_slot_values(self, alignment):
        for group_name in self.groups:
            for group in self.groups[group_name]:
                for slot_name in ontology.high_level_slot_names:
                    if slot_name not in group.slots:
                        continue

                    new_slot_values = []

                    for slot_value in group.slots[slot_name]:
                        new_slot_value = SlotValue()

                        if slot_value.string in alignment:
                            new_slot_value.string = alignment[slot_value.string]
                        else:
                            new_slot_value = slot_value

                        new_slot_values.append(new_slot_value)

                    group.slots[slot_name] = new_slot_values


class Abstract:
    def __init__(self):
        self.abstract_id = None
        self.tokenization = None
        self.annotated_abstract = None
        self.group_collection = None
        self.annotated_abstract_predicted = None

    def import_data(self, disease_string, abstract_id, path):
        base_filename = path + disease_string + ' ' + abstract_id + '_'
        self.abstract_id = abstract_id

        self.tokenization = Tokenization()
        self.tokenization.set_from_file(base_filename + 'export.csv')

        self.annotated_abstract = AnnotatedAbstract()
        self.annotated_abstract.set_from_file(base_filename + 'admin.annodb', self.tokenization)
        # self.annotated_abstract.shrink()

        self.group_collection = GroupCollection()
        self.group_collection.import_group_ids(base_filename + 'admin.n-triples')
        self.group_collection.remove_evidence_quality_groups()
        self.group_collection.fill(self.annotated_abstract)
        # self.group_collection.remove_empty_groups()

    def get_sentence_numbers(self):
        sentence_numbers = set()

        for annotation in self.annotated_abstract.annotations:
            sentence_numbers.add(annotation.sentence_number)

        return sentence_numbers

    def get_annotations_sentence_numbers(self):
        sentence_numbers = []

        for annotation in self.annotated_abstract.annotations:
            sentence_numbers.append(annotation.sentence_number)

        return sentence_numbers

    def get_sentence_tokens(self, sentence_number):
        for annotation in self.annotated_abstract.annotations:
            if annotation.sentence_number == sentence_number:
                return annotation.left_context + annotation.tokens + annotation.right_context

        return None

    def get_used_annotations_of_group_type(self, group_name):
        indices = set()

        if group_name not in self.group_collection.groups:
            return set()

        for group in self.group_collection.groups[group_name]:
            indices.update(group.get_entities_as_global_annotation_indices())

        return indices

    def remove_annotation(self, global_annotation_index):
        # get index with annotation list in annotated_abstract
        index = None

        for i, annotation in enumerate(self.annotated_abstract.annotations):
            if annotation.global_index == global_annotation_index:
                index = i
                break

        if index != None:
            del self.annotated_abstract.annotations[index]

    def remove_inner_annotations(self):
        sentence_numbers = self.get_sentence_numbers()

        for sentence_number in sentence_numbers:
            for label in ontology.used_most_general_classes:
                while self.contains_sentence_overlapping_annotations(sentence_number, label):
                    ranges = self.annotated_abstract.get_annotation_ranges(sentence_number, label)

                    for i in range(len(ranges)):
                        for j in range(i + 1, len(ranges)):
                            start1, end1 = ranges[i][1], ranges[i][2]
                            start2, end2 = ranges[j][1], ranges[j][2]

                            if do_ranges_overlap((start1, end1), (start2, end2)):
                                length1 = end1 - start1
                                length2 = end2 - start2

                                if length1 < length2:
                                    self.remove_annotation(ranges[i][0])
                                else:
                                    self.remove_annotation(ranges[j][0])

    def contains_sentence_overlapping_annotations(self, sentence_number, label):
        ranges = self.annotated_abstract.get_annotation_ranges(sentence_number, label)

        for i in range(len(ranges)):
            for j in range(i + 1, len(ranges)):
                start1, end1 = ranges[i][1], ranges[i][2]
                start2, end2 = ranges[j][1], ranges[j][2]

                if do_ranges_overlap((start1, end1), (start2, end2)):
                    return True

    def contains_overlapping_annotations(self):
        sentence_numbers = self.get_sentence_numbers()

        for label in ontology.used_most_general_classes:
            for sentence_number in sentence_numbers:
                if self.contains_sentence_overlapping_annotations(sentence_number, label):
                    return True

        return False


def evaluate_group_collection_prediction(group_collection_gt, group_collection_predicted):
    f1_statistics_collection = F1StatisticsCollection()

    for group_name in group_collection_gt.groups:
        if len(group_collection_gt.groups[group_name]) != len(group_collection_predicted.groups[group_name]):
            raise ('Unequal number of groups!')

        for i in range(len(group_collection_gt.groups[group_name])):
            group_gt = group_collection_gt.groups[group_name][i]
            group_predicted = group_collection_predicted.groups[group_name][i]
            f1_statistics_collection.update(group_gt, group_predicted)

    return f1_statistics_collection


def get_disease_ids(disease_string, path):
    ids = set()

    if disease_string != 'gl' and disease_string != 'dm2':
        raise ('Invalid disease string!')

    for filename in os.listdir(path):
        space_index = filename.index(' ')
        underscore_index = filename.index('_')
        if filename[:space_index] != disease_string or 'copy' in filename:
            continue

        ids.add(filename[space_index + 1:underscore_index])

    return ids


def import_abstracts(disease_string, path):
    abstracts = []
    ids = get_disease_ids(disease_string, path)

    for abstract_id in ids:
        abstract = Abstract()
        abstract.abstract_id = abstract_id
        abstract.import_data(disease_string, abstract_id, path)
        abstracts.append(abstract)

    return abstracts


def create_dataset_split(dataset_list, validation_fraction, test_fraction):
    random.shuffle(dataset_list)
    original_length = len(dataset_list)

    num_test = int(len(dataset_list) * test_fraction)
    test = dataset_list[:num_test]
    dataset_list = dataset_list[num_test:]

    num_validation = int(validation_fraction * len(dataset_list))
    validation = dataset_list[:num_validation]
    train = dataset_list[num_validation:]

    assert len(test) + len(validation) + len(train) == original_length
    return (train, validation, test)


def set_to_string(s):
    string = '['
    for i, element in enumerate(s):
        string += "'"
        string += element
        string += "'"

        if i < len(s) - 1:
            string += ', '

    string += ']'
    return string


slot_list = ['hasDoseValue', 'hasDrug', 'hasCTDesign', 'hasAuthor', 'hasChangeValue']
vis_slot_name = 'hasDrug'


def evaluate_naive(test_abstracts, disease_string):
    stats_collection = F1StatisticsCollection()

    for i, abstract in enumerate(test_abstracts):
        # print('abstract ' + abstract.abstract_id + ' --------------------------')
        annotated_abstract = abstract.annotated_abstract
        group_collection_gt = abstract.group_collection
        group_collection_predicted = group_collection_gt.create_structure_copy()

        for group_name in group_collection_gt.groups:

            for i in range(len(group_collection_gt.groups[group_name])):
                group_gt = group_collection_gt.groups[group_name][i]
                group_predicted = group_collection_predicted.groups[group_name][i]

                for slot_name in ontology.slot_type:
                    if slot_name not in ontology.used_slots:
                        continue

                    key_annotations = group_gt.get_key_annotations_fixed(annotated_abstract, disease_string, slot_name)

                    if len(key_annotations) == 0:
                        continue
                        # value_annotation = random.choice(annotated_abstract.annotations)
                    else:
                        key_annotation = random.choice(key_annotations)
                        value_annotation = annotated_abstract.get_closest_annotation(key_annotation.doc_char_onset,
                                                                                     slot_name)
                    # create and add slot value
                    if value_annotation != None:
                        slot_value = SlotValue()
                        slot_value.string = ' '.join(value_annotation.tokens)
                        slot_value.annotation = value_annotation
                        group_predicted.add_value(slot_name, slot_value)

                stats_collection.update(group_gt, group_predicted)

                '''
                for slot_name in slot_list:
                    if slot_name not in group_gt.slots:
                        continue
                    
                    print('slot type: ' + slot_name)
                    print('predicted: ' + set_to_string(group_predicted.slot_values_as_set(slot_name)))
                    print('ground truth: ' + set_to_string(group_gt.slot_values_as_set(slot_name)))
                    print(' ')
                '''
    return stats_collection


'''
f = open('glaucoma_slot_filling.dump', 'rb')
train,validation,test = pickle.load(f)
f.close()
sys.exit()
'''

'''
abstracts = import_abstracts('gl', 'data3/')
split = create_dataset_split(abstracts, 0.2, 0.2)
f = open('glaucoma_slot_filling.dump', 'wb')
pickle.dump(split, f)
f.close()
sys.exit()
'''

'''
encoding = abstracts[0].group_collection.encode(abstracts[0].annotated_abstract)
predicted_group_collection = GroupCollection()
predicted_group_collection.set_from_encoded_groups(encoding, abstracts[0].annotated_abstract)
print(evaluate_group_collection_prediction(abstracts[0].group_collection, predicted_group_collection).print_statistics())
'''


def import_individuals(filename):
    f = open(filename)
    f.readline()
    individuals = set()

    for line in f:
        line = line.strip()
        cols = line.split('\t')

        if cols[1] == 'TRUE':
            individuals.add(cols[0])

    return individuals
