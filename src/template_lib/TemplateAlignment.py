from typing import Iterable

from scipy import optimize

from .SlotFillingEvaluation import *


def pad_template_list(list1, list2):
    len_list1 = len(list1)
    len_list2 = len(list2)

    # if both lists have equal length, nothing needs to be done
    if len_list1 == len_list2:
        return

    # add None entries to shorter list s.th. both list have equal length
    shorter_list = list1 if len_list1 < len_list2 else list2

    # estimate number of padding templates neede
    num_padding_temps = max(len_list1, len_list2) - len(shorter_list)

    # add padding templates to shorter list
    shorter_list += [None] * num_padding_temps


def align_templates(
        gt_templates: Iterable[Template],
        predicted_templates: Iterable[Template],
        template_alignment,
        used_slots: Iterable[str] = None
):
    pad_template_list(gt_templates, predicted_templates)
    num_temps = len(gt_templates)

    # create cost matrix
    cost_matrix = np.zeros((num_temps, num_temps))
    for gt_index in range(num_temps):
        for predicted_index in range(num_temps):
            evaluation = SlotFillingEvaluation()
            evaluation.update(
                gt_template=gt_templates[gt_index],
                predicted_template=predicted_templates[predicted_index],
                used_slots=used_slots,
                template_alignment=template_alignment
            )

            # use negative f1 score as cost
            cost_matrix[gt_index, predicted_index] = -evaluation.compute_micro_stats().f1()

    # solve linear assignment problem
    optimal_gt_indices, optimal_predicted_indices = optimize.linear_sum_assignment(cost_matrix)

    # create list of aligned template pairs
    aligned_templates = list()
    for i in range(num_temps):
        aligned_templates.append(
            (gt_templates[optimal_gt_indices[i]], predicted_templates[optimal_predicted_indices[i]]))

    return aligned_templates


class TemplateAlignment:
    def __init__(
            self,
            gt_temp_collection=None,
            predicted_temp_collection=None,
            used_slots=None,
            template_types_ordering: Iterable[str] = None
    ):
        self._used_slots = used_slots
        self._aligned_templates = list()

        if gt_temp_collection is None or predicted_temp_collection is None:
            return

        if template_types_ordering is not None:
            all_temp_types = template_types_ordering
        else:
            all_temp_types = gt_temp_collection.get_template_types() | predicted_temp_collection.get_template_types()

        # align templates for each template type
        for temp_type in all_temp_types:
            # get ground truth and predicted templates for current type
            gt_temps = gt_temp_collection.get_templates_by_type(temp_type)
            predicted_temps = predicted_temp_collection.get_templates_by_type(temp_type)

            # align ground truth and predicted templates
            self._aligned_templates.extend(align_templates(
                gt_templates=gt_temps,
                predicted_templates=predicted_temps,
                used_slots=used_slots,
                template_alignment=self
            ))

    def add_alignment(self, ground_truth_template: Template, predicted_template: Template):
        if ground_truth_template is not None:
            if not isinstance(ground_truth_template, Template):
                raise TypeError('template instance expected')
        if predicted_template is not None:
            if not isinstance(predicted_template, Template):
                raise TypeError('template instance expected')

        self._aligned_templates.append((ground_truth_template, predicted_template))

    def __iter__(self):
        return iter(self._aligned_templates)

    def get_assigned_ground_truth_template(self, predicted_template_id):
        for groud_truth_template, predicted_template in self:
            if groud_truth_template is not None and predicted_template is not None:
                if predicted_template.get_id() == predicted_template_id:
                    return groud_truth_template

        # no ground truth template was assigned to predicted one
        return None

    def print_out(self):
        for gt_template, predicted_template in self:
            print(gt_template, predicted_template)

    def update_evaluation(self, evaluation):
        # iterate template pairs and update evaluation statistics
        for gt_temp, predicted_temp in self:
            evaluation.update(gt_temp, predicted_temp, self, self._used_slots)

        return evaluation
