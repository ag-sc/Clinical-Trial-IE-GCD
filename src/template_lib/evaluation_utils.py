from typing import Dict, List, Tuple
from itertools import chain
from template_lib.ctro import CtroOntology
from template_lib.SlotFillingEvaluation import SlotFillingEvaluation
from template_lib.data_classes.TemplateCollection import TemplateCollection
from template_lib import consts


class TemplateCardinalityPredictionEvaluation:
    def __int__(self):
        self.ground_truth_cardinalities: Dict[str, List[int]] = dict()
        self.predicted_cardinalities: Dict[str, List[int]] = dict()
        self.abs_cardinality_diff: Dict[str, List[int]] = dict()

    def add_cardinality_prediction(
            self,
            template_name: str,
            ground_truth_cardinality: int,
            predicted_cardinality: int
    ):
        if template_name not in self.ground_truth_cardinalities:
            self.ground_truth_cardinalities[template_name] = list()
        if template_name not in self.predicted_cardinalities:
            self.predicted_cardinalities[template_name] = list()
        if template_name not in self.abs_cardinality_diff:
            self.abs_cardinality_diff[template_name] = list()

        self.ground_truth_cardinalities[template_name].append(ground_truth_cardinality)
        self.predicted_cardinalities[template_name].append(predicted_cardinality)
        self.abs_cardinality_diff[template_name].append(abs(predicted_cardinality - ground_truth_cardinality))

    def update(
            self,
            ground_truth_template_collection: TemplateCollection,
            predicted_template_collection: TemplateCollection
    ):
        ctro_ontology = CtroOntology(consts.template_lib_path)
        for template_name in ctro_ontology.group_names:
            gt_templates = ground_truth_template_collection.get_templates_by_type(template_name)
            predicted_templates = predicted_template_collection.get_templates_by_type(template_name)
            self.add_cardinality_prediction(
                template_name=template_name,
                ground_truth_cardinality=len(gt_templates),
                predicted_cardinality=len(predicted_templates)
            )


def compute_mean_f1_over_templates(slot_filling_evaluation: SlotFillingEvaluation) -> Tuple[List[str], List[float]]:#Dict[str, float]:
    ctro_ontology = CtroOntology(consts.template_lib_path)
    all_used_slots = slot_filling_evaluation.keys()
    f1_list_per_template = dict()

    for template_name in ctro_ontology.group_slots:
        used_template_slots = ctro_ontology.group_slots[template_name] & all_used_slots
        if len(used_template_slots) == 0:
            continue
        f1_list_per_template[template_name] = [slot_filling_evaluation[slot_name].f1() for slot_name in used_template_slots]

    tns = []
    f1s = []
    for tn in f1_list_per_template:
        tns.append(tn)
        f1s.append(sum(f1_list_per_template[tn]) / len(f1_list_per_template[tn]))
    return tns, f1s

    # return {
    #     template_name: sum(f1_list_per_template[template_name]) / len(f1_list_per_template[template_name])
    #     for template_name in f1_list_per_template
    # }
