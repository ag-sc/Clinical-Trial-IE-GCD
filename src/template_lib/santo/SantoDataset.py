import os.path
from typing import Iterable

from template_lib.santo.SantoAnnotationFile import SantoAnnotationFile
from template_lib.santo.SantoTokenizationFile import SantoTokenizationFile
from template_lib.santo.SantoTriplesFile import *


def import_trial_ids(ids_filename):
    with open(ids_filename) as f:
        return {line.strip() for line in f}


def create_santo_filename_prefix(disease, annotator_name, trial_id):
    return disease + ' ' + str(trial_id) + '_' + annotator_name


def link_templates(
        template_collection: TemplateCollection,
        triples_file: SantoTriplesFile,
        slots_containing_templates: Iterable[str],
) -> TemplateCollection:
    for subject, predicate, object in triples_file:
        if is_rdf_type_predicate(predicate) or is_rdf_label_predicate(predicate):
            continue

        slot_name = extract_rdf_identifier(predicate)
        if slot_name not in slots_containing_templates:
            continue

        subject_template_id = extract_rdf_identifier(subject)
        object_template_id = extract_rdf_identifier(object)

        try:
            subject_template = template_collection.get_template_by_id(subject_template_id)
            object_template = template_collection.get_template_by_id(object_template_id)
            subject_template.add_slot_filler(slot_name, object_template)
        except KeyError as key_error:
            # TODO error handling, logging
            pass
    return template_collection


def import_santo_document(path, disease, annotator_name, _id, tokenizer=None):
    # import tokenization data
    filename_prefix = os.path.join(path, create_santo_filename_prefix(disease, 'export', _id))
    tokenization_file = SantoTokenizationFile(filename_prefix + '.csv')

    # triples and annotation file use same filename prefix
    filename_prefix = os.path.join(path, create_santo_filename_prefix(disease, annotator_name, _id))

    # import santo annotation file
    annotation_file = SantoAnnotationFile(filename_prefix + '.annodb')

    # import santo triples file
    triples_file = SantoTriplesFile(filename_prefix + '.n-triples')

    # extract sentence objects from annotation file
    sentences = tokenization_file.extract_all_sentences(tokenizer)

    # use annotation file to extract entities of sentence and add entities
    # to sentence object
    for sentence in sentences:
        entities = annotation_file.extract_entities(sentence.get_index() + 1, tokenization_file, tokenizer)

        for entity in entities:
            sentence.add_entity(entity)

    # create final document
    document = Document(sentences)
    document.set_id(_id)

    # create template collection
    entities = document.get_entities()
    template_collection = triples_file.extract_initial_template_collection()
    template_collection.assign_entities(entities)

    link_templates(
        template_collection=template_collection,
        triples_file=triples_file,
        slots_containing_templates=[
            'describes',
            'hasArm',
            'hasPopulation',
            'hasDiffBetweenGroups',
            'hasOutcome1',
            'hasOutcome2',
            'hasOutcome',
            'hasAdverseEffect',
            'hasIntervention',
            'hasMedication',
            'hasEndpoint'
        ]
    )

    # remove templates which ae not referenced by any other templates
    # exception: Publication template as this template is always root node
    '''
    while True:
        empty_template_found = False

        for template_id, ref_count in template_collection.count_template_references().items():
            if ref_count == 0:
                if template_collection[template_id].get_type() != 'Publication':
                    template_collection.remove(template_id)
                    empty_template_found = True

        if not empty_template_found:
            break
    '''
    return document, template_collection


class SantoDataset:
    def __init__(self,
                 directory_path=None,
                 trial_ids=None,
                 disease=None,
                 annotator_name=None,
                 tokenizer=None,
                 data=None):
        if data is None:
            # data is list of pairs (document, template_collection)
            self._data = [import_santo_document(directory_path, disease, annotator_name, trial_id, tokenizer)
                          for trial_id in sorted(trial_ids)]
        else:
            self._data = data

    def to_dict(self, simplified=False):
        if simplified:
            return sum([
                template_collection.to_dict(simplified=simplified)
                for _, template_collection in self._data
            ], [])
        else:
            return {
                "SantoDataset": {
                    "documents": [
                        document.to_dict(simplified=simplified)
                        for document, _ in self._data
                    ],
                    "template_collections": [
                        template_collection.to_dict(simplified=simplified)
                        for _, template_collection in self._data
                    ],
                }
            }

    def to_json(self, simplified=False):
        return json.dumps(self.to_dict(simplified=simplified), indent=2)

    @classmethod
    def from_dict(cls, d):
        assert len(d) == 1 and "SantoDataset" in d
        d = d["SantoDataset"]
        ds = cls(data=[
            (Document.from_dict(d["documents"][i]), TemplateCollection.from_dict(d["template_collections"][i]))
            for i in range(len(d["documents"]))
        ])
        return ds

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(json.loads(s))

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, doc_id):
        for document, template_collection in self:
            if document.get_id() == doc_id:
                return document, template_collection

        # document was not found -> raise KeyError
        raise KeyError('no document found with id ' + str(doc_id))
