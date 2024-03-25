import csv
import json
import os
import pickle
import re
import statistics
from argparse import ArgumentParser
from collections import defaultdict
from pprint import pprint
from typing import Optional, List

import pandas as pd
import torch

import generative_approach.data_handling as gen_dh
import generative_approach.evaluation as gen_eval
from generative_approach.models.flan_t5 import FlanT5
from generative_approach.template_coding import LinearTemplateEncoder
from generative_approach.utils import import_task_config, get_model_class, TemplateGenerationTokenizer
from template_lib import evaluation_utils
from template_lib.data_classes.Template import Template
from template_lib.data_classes.TemplateCollection import TemplateCollection
from template_lib.data_handling import import_train_test_valid_data_from_task_config
from template_lib.max_cardinalities_estimator import gen_max_slot_cardinalities

def mean_dataframe(dfs: List[pd.DataFrame]):
    assert len(dfs) > 0
    dfs.sort(key=lambda x: x.shape[0], reverse=True)
    col0 = dfs[0].iloc[:, 0].tolist()

    row_ids = {col0[i]: i for i in range(len(col0))}

    assert all([len(set(df.iloc[:, 0].tolist()).difference(col0)) == 0 for df in dfs])
    col_nums = [df.shape[1] for df in dfs]
    assert len(set(col_nums)) == 1

    if len(dfs) == 1:
        return dfs[0]
    else:
        data = [
            [
                list()
                for j in range(len(col0))
            ]
            for i in range(col_nums[0]-1)
        ]
        for df in dfs:
            for _, row in df.iterrows():
                row = row.tolist()
                for col_id in range(1, len(row)):
                    data[col_id-1][row_ids[row[0]]].append(row[col_id])
            #for i in range(col_nums[0]-1):
            #    for j in range(len(col0)):
            #        data[i][j].append(df.iloc[i+1, j])


        data_res = [
            [
                f"{round(statistics.mean(data[i][j]), 2)} ($\\pm$ {round(statistics.stdev(data[i][j]), 2)})" if " GT " not in dfs[0].columns[i+1] else round(statistics.median(data[i][j]), 2)
                for j in range(len(col0))
            ]
            for i in range(col_nums[0]-1)
        ]
        data_res = [col0] + data_res
        return pd.DataFrame.from_dict({
            col: col_data
            for col, col_data in zip(dfs[0].columns, data_res)
        })


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--results', required=True)
    # argparser.add_argument('--nosloteval', action='store_false')
    # argparser.add_argument('--noinstancecount', action='store_false')
    arguments = argparser.parse_args()

    results_path = arguments.results
    run_best_eval = True  # arguments.nosloteval
    # run_instance_count = arguments.noinstancecount

    names = {
        "gen": ["t5", "led"],
    }

    name_res = {
        "gen": re.compile(r"^(?P<path>.*)/gen_model_(?P<modeltype>.*)_(?P<disease>.*)_(?P<model>.*)_(?P<date>.*)_(?P<time>.*)_(?P<batchsize>.*)_(?P<learningrate>.*)_(?P<lambda>.*)$"),
    }

    modeltypes = {
        "gen": ["basic", "ptrmodel"],
    }

    diseases = ["dm2", "gl"]

    besteval_path = os.path.join(results_path, "best_res_sloteval.pkl")
    # instancecount_path = os.path.join(results_path, "best_res_instancecount.pkl")

    best_res_evals = {
        x: {
            y: {
                z: {
                    w: {
                        v: None
                        for v in names[y]
                    }
                    for w in (["", "nodec"] if y == "gen" else [""])
                }
                for z in modeltypes[y]
            }
            for y in names.keys()
        }
        for x in diseases
    }

    if os.path.isfile(besteval_path):
        with open(besteval_path, "rb") as sres_pkl_file:
            best_res_evals = pickle.load(sres_pkl_file)
            total_best = []
            avg_slot_f1_best = []
            count_best = []

            # total_best_labels = []
            for disease in diseases:
                for approach in best_res_evals[disease]:
                    for modeltype in best_res_evals[disease][approach]:
                        for nd in best_res_evals[disease][approach][modeltype]:
                            modelres = [([w.get_dataframe_f1(col_name=f"{disease} {approach} {modeltype} {'GCD' if len(nd) == 0 else 'noGCD'} $F_1$") for w in v], v, model) for model, v in best_res_evals[disease][approach][modeltype][nd].items() if v is not None]
                            if len(modelres) == 0:
                                continue
                            modelres.sort(key=lambda x: statistics.mean([w.iloc[-1].tolist()[1] for w in x[0]]), reverse=True)
                            # [best_res_sloteval[k][approach][model] for model in best_res_sloteval[k][approach]]
                            meandf = mean_dataframe(modelres[0][0])
                            total_best.append(meandf)
                            # total_best_labels.append((disease, approach))

                            print("\\subsection{Count eval " + disease + " " + approach + " " + modeltype+" "+('GCD' if len(nd) == 0 else 'noGCD')+"}")
                            count_meandf = mean_dataframe([w.get_count_dataframe() for w in modelres[0][1]])
                            print(count_meandf.round(2).to_latex(index=False, multicolumn_format="c", longtable=True,
                                                                 na_rep="-", float_format="%.2f"))

                            diffcount_meandf = mean_dataframe([w.get_diff_count_dataframe(col_name=f"{disease} {approach} {modeltype} {'GCD' if len(nd) == 0 else 'noGCD'} Mean Abs Diff") for w in modelres[0][1]])
                            count_best.append(diffcount_meandf)


                            tns_f1s = [evaluation_utils.compute_mean_f1_over_templates(w) for w in modelres[0][1]]
                            slot_f1_dfs = [pd.DataFrame.from_dict({"Template Name": tns, f"{disease} {approach} {modeltype} {'GCD' if len(nd) == 0 else 'noGCD'} $F_1$": f1s}) for tns, f1s in tns_f1s]
                            #tns, f1s = evaluation_utils.compute_mean_f1_over_templates(modelres[0][1])

                            avg_slot_f1 = mean_dataframe(slot_f1_dfs)

                            avg_slot_f1_best.append(avg_slot_f1)

                            # print(evaluation_utils.compute_mean_f1_over_templates(modelres[0][1]))

            joined = total_best[0]
            for df in total_best[1:]:
                joined = pd.merge(joined, df, on="Slot Name", how="outer")
            #joined["Mean $F_1$"] = joined.mean(numeric_only=True, axis=1)
            #joined.sort_values(by="Mean $F_1$", inplace=True, ascending=False)
            joined.sort_values(by="Slot Name", inplace=True, ascending=True)
            print("### Joined slot eval")
            print(joined.round(2).to_latex(index=False, multicolumn_format="c", longtable=True, na_rep="-",
                                           float_format="%.2f"))

            joined2 = avg_slot_f1_best[0]
            for df in avg_slot_f1_best[1:]:
                joined2 = pd.merge(joined2, df, on="Template Name", how="outer")

            print("### Joined avg slot f1 eval")
            joined2.sort_values(by="Template Name", inplace=True, ascending=True)
            print(joined2.round(2).to_latex(index=False, multicolumn_format="c", longtable=True, na_rep="-",
                                            float_format="%.2f"))

            joined3 = count_best[0]
            for df in count_best[1:]:
                joined3 = pd.merge(joined3, df, on="Template Name", how="outer")

            print("### Joined count eval")
            joined3.sort_values(by="Template Name", inplace=True, ascending=True)
            print(joined3.round(2).to_latex(index=False, multicolumn_format="c", longtable=True, na_rep="-",
                                            float_format="%.2f"))

            run_best_eval = False

    with pd.ExcelWriter(os.path.join(results_path, 'summary.xlsx'), engine='xlsxwriter') as writer:
        workbook = writer.book

        for approach in names.keys():
            row_headings = ["Model", "Model Type", "Mean $F_1$ ($\\pm \\sigma$)"]

            summarydf = pd.DataFrame(columns=pd.MultiIndex.from_arrays([sum([[d, d, d] for d in diseases], []),
                                                                        row_headings * len(diseases)]))
            # ["model", "f1"]*len(diseases))

            for modeltype in modeltypes[approach]:

                nodec = [""]
                if approach == "gen":
                    nodec = ["", "nodec"]

                for nd in nodec:

                    for model in names[approach]:
                        row = []
                        for disease in diseases:
                            print(f"{approach}-{disease}-{model}-{modeltype}-{'GCD' if len(nd) == 0 else 'noGCD'}")
                            csv_files = [os.path.join(results_path, approach, modeltype, f"{disease}-{model}", name)
                                         for name in os.listdir(os.path.join(results_path, approach, modeltype, f"{disease}-{model}"))
                                         if os.path.isfile(os.path.join(results_path, approach, modeltype, f"{disease}-{model}", name))
                                         and name.endswith(nd + "-stats.csv") and (len(nd) > 0 or not name.endswith("-nodec-stats.csv"))]

                            config_files = [os.path.join(results_path, approach, modeltype, f"{disease}-{model}", name)
                                            for name in os.listdir(os.path.join(results_path, approach, modeltype, f"{disease}-{model}"))
                                            if os.path.isfile(os.path.join(results_path, approach, modeltype, f"{disease}-{model}", name))
                                            and name.startswith("config_") and name.endswith(".json")]

                            stats_files = [os.path.join(results_path, approach, modeltype, f"{disease}-{model}", name)
                                           for name in os.listdir(os.path.join(results_path, approach, modeltype, f"{disease}-{model}"))
                                           if os.path.isfile(os.path.join(results_path, approach, modeltype, f"{disease}-{model}", name))
                                           and name.endswith(nd + "-stats.pkl") and (len(nd) > 0 or not name.endswith("-nodec-stats.pkl"))]

                            if len(csv_files) == 0 and len(config_files) == 0 and len(stats_files) == 0:
                                print(f"No files found for {approach}-{disease}-{model}-{modeltype}")
                                continue

                            # print(csv_files)
                            assert len(csv_files) >= 1
                            assert len(config_files) == 1
                            assert len(stats_files) >= 1
                            csv_files.sort(reverse=True)  # newest files to top
                            stats_files.sort(reverse=True)  # newest files to top
                            csv_file_path = csv_files[0]
                            config_file_path = config_files[0]
                            stats_file_path = stats_files[0]

                            with open(stats_file_path, "rb") as fp:
                                stats_dicts = pickle.load(fp)

                            worksheet = workbook.add_worksheet(f"{approach}-{disease}-{model}-{modeltype}-{'GCD' if len(nd) == 0 else 'noGCD'}")
                            writer.sheets[f"{approach}-{disease}-{model}-{modeltype}-{'GCD' if len(nd) == 0 else 'noGCD'}"] = worksheet

                            df = pd.read_csv(csv_file_path)
                            df.sort_values(by=["validation-f1"], inplace=True, ascending=False)
                            df.reset_index(drop=True, inplace=True)

                            model_path = df.iloc[0]["model"]

                            match = re.search(name_res[approach], df.iloc[0]["model"])

                            # "gen": re.compile(r"^(?P<path>.*)/gen_model_(?P<disease>.*)_(?P<model>.*)_(?P<date>.*)_(?P<time>.*)_(?P<batchsize>.*)_(?P<learningrate>.*)_(?P<lambda>.*)$"),
                            # "extr": re.compile(r"^(?P<path>.*)/extr_model_(?P<disease>.*)_(?P<model>.*)_(?P<date>.*)_(?P<time>.*)_(?P<batchsize>.*)_(?P<learningrate>.*)_(?P<lambda>.*).pt$")

                            mean = round(df["test-f1"].mean(), 3)
                            stddev = round(df["test-f1"].std(), 3)
                            row.append(match.group("model"))
                            row.append(modeltype+nd)
                            #row.append(mean)
                            row.append(f"${mean:.3f}~(\\pm {stddev:.3f})$")

                            df.to_excel(writer, sheet_name=f"{approach}-{disease}-{model}-{modeltype}-{'GCD' if len(nd) == 0 else 'noGCD'}", startrow=0, startcol=0)
                            print(f"### Eval results {approach}-{disease}-{model}-{modeltype}-{'GCD' if len(nd) == 0 else 'noGCD'}")
                            print(df.round(2).to_latex(index=False, multicolumn_format="c", float_format="%.2f"))
                            # , column_format="c"

                            best_res_evals[disease][approach][modeltype][nd][model] = [i["res_test"] for i in stats_dicts]

                            for column in df:
                                column_length = max(df[column].astype(str).map(len).max(), len(column))
                                col_idx = df.columns.get_loc(column)
                                writer.sheets[f"{approach}-{disease}-{model}-{modeltype}-{'GCD' if len(nd) == 0 else 'noGCD'}"].set_column(col_idx + 1, col_idx + 1, column_length)

                        # with open(csv_file_path) as csv_file:
                        #     csv_dict = csv.DictReader(csv_file, delimiter=',')
                        #     for row in csv_dict:
                        #         print(row)

                        if len(row) > 0:
                            summarydf.loc[len(summarydf)] = row
            print("### Eval summary " + approach)
            print(summarydf.round(2).to_latex(index=False, multicolumn_format="c",
                                              float_format="%.2f"))  # , column_format="c"
            sworksheet = workbook.add_worksheet(f"summary-{approach}")
            writer.sheets[f"summary-{approach}"] = sworksheet
            summarydf.to_excel(writer, sheet_name=f"summary-{approach}", startrow=0, startcol=0)

            for column in summarydf:
                column_length = max(summarydf[column].astype(str).map(len).max(), len(column))
                col_idx = summarydf.columns.get_loc(column)
                writer.sheets[f"summary-{approach}"].set_column(col_idx + 1, col_idx + 1, column_length)

    if run_best_eval:
        print("### Best results raw")
        print(best_res_evals)
        with open(besteval_path, "wb") as sres_pkl_file:
            pickle.dump(best_res_evals, sres_pkl_file)
