import statistics
from collections import defaultdict

from Levenshtein import distance as levenshtein_distance

import pandas as pd

from template_lib.data_classes.Template import *
from .ctro import *
from .data_classes.TemplateCollection import TemplateCollection


class SlotFillingEvaluation:
    def __init__(self):
        # dict for F1Statistics
        # key: slot name
        # value F1Statistics instance
        self._statistics = dict()
        # key: template name
        # value: list of pairs (gt number of instances, predicted number of instances)
        self._instance_counts = defaultdict(list)

    # def __getitem__(self, key):
    #     return self._statistics[key]

    @staticmethod
    def normalized_sim(string1, string2):
        return 1 - levenshtein_distance(string1, string2) / max(len(string1), len(string2))

    def update_instance_counts(self,
                               gt_template_collection: TemplateCollection,
                               predicted_template_collection: TemplateCollection):
        gt = gt_template_collection.grouped_by_type
        pred = predicted_template_collection.grouped_by_type
        template_names = set(gt.keys()).union(pred.keys())
        for tn in template_names:
            gt_count = len(gt[tn]) if tn in gt else 0
            pred_count = len(pred[tn]) if tn in pred else 0
            self._instance_counts[tn].append((gt_count, pred_count))

    def update(self, gt_template, predicted_template, template_alignment=None, used_slots=None):
        # replace None argument by empty template
        # only one argumnet can be None
        if gt_template is None and predicted_template is None:
            raise Exception('only one argument can be None')
        if gt_template is None:
            gt_template = Template(None, None)
        if predicted_template is None:
            predicted_template = Template(None, None)

        # compute union of slot names used in ground truth and predicted template
        slot_names_union = gt_template.get_filled_slot_names() | predicted_template.get_filled_slot_names()

        # update f1 statistics for each slot name
        for slot_name in slot_names_union:
            if used_slots is not None and slot_name not in used_slots:
                continue
            if slot_name in self._statistics:
                f1_stats = self._statistics[slot_name]
            else:
                f1_stats = F1Statistics()
                self._statistics[slot_name] = f1_stats

            # get ground truth and predicted slot fillers as strings
            slot_fillers_gt = set(gt_template.get_slot_fillers_as_strings(slot_name))
            slot_fillers_predicted = set()

            for slot_filler in predicted_template.get_slot_fillers(slot_name):
                if isinstance(slot_filler, Entity):
                    slot_fillers_predicted.add(' '.join(slot_filler.get_tokens()))
                elif isinstance(slot_filler, Template):
                    assigned_gt_temp = template_alignment.get_assigned_ground_truth_template(slot_filler.get_id())
                    if assigned_gt_temp is not None:
                        slot_fillers_predicted.add(assigned_gt_temp.get_id())
                    else:
                        slot_fillers_predicted.add(slot_filler.get_id())

            intersec = slot_fillers_gt & slot_fillers_predicted

            # update statistics for slot
            f1_stats.num_occurences += len(slot_fillers_gt)
            f1_stats.true_positives += len(intersec)
            for sfgt in slot_fillers_gt - intersec:
                for sfpred in slot_fillers_predicted - intersec:
                    #TODO: all this omitting/adding spaces thing is a bit hacky
                    if self.normalized_sim(sfgt.replace(" ", ""), sfpred.replace(" ", "")) >= 0.9:
                        f1_stats.true_positives += 1

            f1_stats.false_positives += len(slot_fillers_predicted - slot_fillers_gt)

    def compute_micro_stats(self, slot_names=None):
        micro_stats = F1Statistics()

        # if slot_names parameter is not given, use all slots
        if slot_names is None:
            slot_names = self._statistics.keys()

        # sum up true positives, false positives and num occurences over all slots
        for slot_name, stats in self._statistics.items():
            # check if slot should be considered
            if slot_name in slot_names:
                micro_stats.true_positives += stats.true_positives
                micro_stats.false_positives += stats.false_positives
                micro_stats.num_occurences += stats.num_occurences

        return micro_stats

    def keys(self):
        return set(self._statistics.keys())

    def print_out(self):
        for slot_name in sorted(self._statistics.keys()):
            stats = self._statistics[slot_name]

            print('slot name: {}; precision: {:.2f}; recall: {:.2f}, f1: {:.2f}'.format(slot_name.replace('has', ''),
                                                                                        stats.precision(),
                                                                                        stats.recall(), stats.f1()))

        # micro f1
        micro_stats = self.compute_micro_stats()
        print('micro stats: precision: {:.2f}; recall: {:.2f}, f1: {:.2f}'.format(micro_stats.precision(),
                                                                                  micro_stats.recall(),
                                                                                  micro_stats.f1()))

    def get_dataframe(self):
        df = pd.DataFrame(columns=["Slot Name", "Precision", "Recall", "$F_1$"])
        for slot_name in sorted(self._statistics.keys()):
            stats = self._statistics[slot_name]
            df.loc[len(df)] = [slot_name.replace('has', ''), stats.precision(), stats.recall(), stats.f1()]
        micro_stats = self.compute_micro_stats()
        df.loc[len(df)] = ["micro", micro_stats.precision(), micro_stats.recall(), micro_stats.f1()]
        return df

    def get_dataframe_f1(self, col_name="$F_1$"):
        df = pd.DataFrame(columns=["Slot Name", col_name])
        for slot_name in sorted(self._statistics.keys()):
            stats = self._statistics[slot_name]
            df.loc[len(df)] = [slot_name.replace('has', ''), stats.f1()]
        micro_stats = self.compute_micro_stats()
        df.loc[len(df)] = ["micro", micro_stats.f1()]
        return df

    def __getitem__(self, slot_name: str):
        return self._statistics[slot_name]

    def get_count_dataframe(self):
        df = pd.DataFrame(columns=["Template Name", "Mean GT Count", "Mean Predicted Count", "Abs Diff"])
        for template_name in sorted(self._instance_counts.keys()):
            stats = self._instance_counts[template_name]
            gt_counts = [x[0] for x in stats]
            pred_counts = [x[1] for x in stats]
            gt_mean = statistics.mean(gt_counts) if len(gt_counts) > 0 else None
            pred_mean = statistics.mean(pred_counts) if len(pred_counts) > 0 else None
            abs_diff = abs(gt_mean - pred_mean) if gt_mean is not None and pred_mean is not None else None

            df.loc[len(df)] = [template_name, gt_mean, pred_mean, abs_diff]
        #micro_stats = self.compute_micro_stats()
        #df.loc[len(df)] = ["micro", micro_stats.f1()]
        return df

    def get_diff_count_dataframe(self, col_name="Abs Diff"):
        df = pd.DataFrame(columns=["Template Name", col_name])
        for template_name in sorted(self._instance_counts.keys()):
            stats = self._instance_counts[template_name]
            gt_counts = [x[0] for x in stats]
            pred_counts = [x[1] for x in stats]
            gt_mean = statistics.mean(gt_counts) if len(gt_counts) > 0 else None
            pred_mean = statistics.mean(pred_counts) if len(pred_counts) > 0 else None
            abs_diff = abs(gt_mean - pred_mean) if gt_mean is not None and pred_mean is not None else None

            df.loc[len(df)] = [template_name, abs_diff]
        return df
