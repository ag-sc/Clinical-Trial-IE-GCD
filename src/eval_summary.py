import csv
import json
import os
import pickle
import re
from argparse import ArgumentParser
from collections import defaultdict
from pprint import pprint
from typing import Optional

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


def escape(s: str):
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%").replace(">", "\\textgreater{}").replace("<",
                                                                                                                 "\\textless{}")


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--results', required=True)
    #argparser.add_argument('--nosloteval', action='store_false')
    #argparser.add_argument('--noinstancecount', action='store_false')
    arguments = argparser.parse_args()

    results_path = arguments.results
    run_best_eval = True #arguments.nosloteval
    #run_instance_count = arguments.noinstancecount

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

    case_study_path = os.path.join(results_path, "case_study_results.pkl")
    besteval_path = os.path.join(results_path, "best_res_sloteval.pkl")
    bestparams_path = os.path.join(results_path, "best_params.pkl")

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

            #total_best_labels = []
            for disease in diseases:
                for approach in best_res_evals[disease]:
                    for modeltype in best_res_evals[disease][approach]:
                        for nd in best_res_evals[disease][approach][modeltype]:
                            modelres = [(v.get_dataframe_f1(col_name=f"{disease} {approach} {modeltype} {'dec' if len(nd) == 0 else nd} $F_1$"), v) for v in best_res_evals[disease][approach][modeltype][nd].values() if v is not None]
                            if len(modelres) == 0:
                                continue
                            modelres.sort(key=lambda x: x[0].iloc[-1].tolist()[1], reverse=True)  # [best_res_sloteval[k][approach][model] for model in best_res_sloteval[k][approach]]
                            total_best.append(modelres[0][0])
                            #total_best_labels.append((disease, approach))

                            print("### Count eval " + disease + " " + approach)
                            print(modelres[0][1].get_count_dataframe().round(2).to_latex(index=False, multicolumn_format="c", longtable=True, na_rep="-", float_format="%.2f"))

                            count_best.append(modelres[0][1].get_diff_count_dataframe(col_name=f"{disease} {approach} {modeltype} {'dec' if len(nd) == 0 else nd} Mean Abs Diff"))

                            tns, f1s = evaluation_utils.compute_mean_f1_over_templates(modelres[0][1])

                            avg_slot_f1 = pd.DataFrame.from_dict({"Template Name": tns, f"{disease} {approach} {modeltype} {'dec' if len(nd) == 0 else nd} $F_1$": f1s})
                            avg_slot_f1_best.append(avg_slot_f1)

                    #print(evaluation_utils.compute_mean_f1_over_templates(modelres[0][1]))

            joined = total_best[0]
            for df in total_best[1:]:
                joined = pd.merge(joined, df, on="Slot Name", how="outer")
            joined["Mean $F_1$"] = joined.mean(numeric_only=True, axis=1)
            joined.sort_values(by="Mean $F_1$", inplace=True, ascending=False)
            print("### Joined slot eval")
            print(joined.round(2).to_latex(index=False, multicolumn_format="c", longtable=True, na_rep="-", float_format="%.2f"))

            joined2 = avg_slot_f1_best[0]
            for df in avg_slot_f1_best[1:]:
                joined2 = pd.merge(joined2, df, on="Template Name", how="outer")

            print("### Joined avg slot f1 eval")
            print(joined2.round(2).to_latex(index=False, multicolumn_format="c", longtable=True, na_rep="-", float_format="%.2f"))

            joined3 = count_best[0]
            for df in count_best[1:]:
                joined3 = pd.merge(joined3, df, on="Template Name", how="outer")

            print("### Joined count eval")
            print(joined3.round(2).to_latex(index=False, multicolumn_format="c", longtable=True, na_rep="-", float_format="%.2f"))

            run_best_eval = False

    best_hyperparams = []


    with pd.ExcelWriter(os.path.join(results_path, 'summary.xlsx'), engine='xlsxwriter') as writer:
        workbook = writer.book

        for approach in names.keys():
            summarydf = pd.DataFrame(columns=pd.MultiIndex.from_arrays([sum([[d, d, d] for d in diseases], []),
                                                                        ["model", "modeltype", "$F_1$"] * len(diseases)]))
            # ["model", "f1"]*len(diseases))
            for modeltype in modeltypes[approach]:
                nodec = [""]
                if approach == "gen":
                    nodec = ["", "nodec"]

                for nd in nodec:

                    for model in names[approach]:
                        row = []
                        for disease in diseases:
                            print(f"{approach}-{disease}-{model}-{modeltype}-{'dec' if len(nd) == 0 else nd}")
                            csv_files = [os.path.join(results_path, approach, modeltype, f"{disease}-{model}", name)
                                         for name in os.listdir(os.path.join(results_path, approach, modeltype, f"{disease}-{model}"))
                                         if os.path.isfile(os.path.join(results_path, approach, modeltype, f"{disease}-{model}", name))
                                         and name.endswith(nd+"-stats.csv") and (len(nd) > 0 or not name.endswith("-nodec-stats.csv"))]

                            config_files = [os.path.join(results_path, approach, modeltype, f"{disease}-{model}", name)
                                            for name in os.listdir(os.path.join(results_path, approach, modeltype, f"{disease}-{model}"))
                                            if os.path.isfile(os.path.join(results_path, approach, modeltype, f"{disease}-{model}", name))
                                            and name.startswith("config_") and name.endswith(".json")]

                            stats_files = [os.path.join(results_path, approach, modeltype, f"{disease}-{model}", name)
                                           for name in os.listdir(os.path.join(results_path, approach, modeltype, f"{disease}-{model}"))
                                           if os.path.isfile(os.path.join(results_path, approach, modeltype, f"{disease}-{model}", name))
                                           and name.endswith(nd+"-stats.pkl") and (len(nd) > 0 or not name.endswith("-nodec-stats.pkl"))]

                            if len(csv_files) == 0 and len(config_files) == 0 and len(stats_files) == 0:
                                print(f"No files found for {approach}-{disease}-{model}-{modeltype}")
                                continue

                            # print(csv_files)
                            assert len(csv_files) >= 1
                            assert len(config_files) == 1
                            assert len(stats_files) >= 1
                            csv_files.sort(reverse=True)# newest files to top
                            stats_files.sort(reverse=True)# newest files to top
                            csv_file_path = csv_files[0]
                            config_file_path = config_files[0]
                            stats_file_path = stats_files[0]
                            # nodec_csv_file_path = csv_files[0] if len(nodec_csv_files) > 0 else None
                            # nodec_stats_file_path = nodec_stats_files[0] if len(nodec_stats_files) > 0 else None


                            with open(stats_file_path, "rb") as fp:
                                stats_dicts = pickle.load(fp)

                            # nodec_stats_dicts = None
                            # if nodec_stats_file_path is not None:
                            #     with open(nodec_stats_file_path, "rb") as fp:
                            #         nodec_stats_dicts = pickle.load(fp)

                            worksheet = workbook.add_worksheet(f"{approach}-{disease}-{model}-{modeltype}-{'dec' if len(nd) == 0 else nd}")
                            writer.sheets[f"{approach}-{disease}-{model}-{modeltype}-{'dec' if len(nd) == 0 else nd}"] = worksheet

                            df = pd.read_csv(csv_file_path)
                            df.sort_values(by=["validation-f1"], inplace=True, ascending=False)
                            df.reset_index(drop=True, inplace=True)

                            model_path = df.iloc[0]["model"]

                            match = re.search(name_res[approach], df.iloc[0]["model"])

                            # "gen": re.compile(r"^(?P<path>.*)/gen_model_(?P<disease>.*)_(?P<model>.*)_(?P<date>.*)_(?P<time>.*)_(?P<batchsize>.*)_(?P<learningrate>.*)_(?P<lambda>.*)$"),
                            # "extr": re.compile(r"^(?P<path>.*)/extr_model_(?P<disease>.*)_(?P<model>.*)_(?P<date>.*)_(?P<time>.*)_(?P<batchsize>.*)_(?P<learningrate>.*)_(?P<lambda>.*).pt$")

                            best_hyperparams.append({
                                "modelpath": df.iloc[0]["model"],
                                "configpath": config_file_path,
                                "disease": disease,
                                "approach": approach,
                                "model": model,
                                "modeltype": modeltype,
                                "batchsize": match.group("batchsize"),
                                "learningrate": match.group("learningrate"),
                                "lambda": match.group("lambda"),
                                "date": match.group("date"),
                                "time": match.group("time"),
                            })

                            # row.append(df.iloc[0]["model"])
                            row.append(match.group("model"))
                            row.append(modeltype+nd)
                            row.append(df.iloc[0]["test-f1"])
                            df.to_excel(writer, sheet_name=f"{approach}-{disease}-{model}-{modeltype}-{'dec' if len(nd) == 0 else nd}", startrow=0, startcol=0)
                            print(f"### Eval results {approach}-{disease}-{model}-{modeltype}-{'dec' if len(nd) == 0 else nd}")
                            print(df.round(2).to_latex(index=False, multicolumn_format="c", float_format="%.2f"))  # , column_format="c"

                            best_stats = [i for i in stats_dicts if i["model"] == df.iloc[0]["model"]][0]

                            #if run_instance_count:
                            #    best_res_instancecount[disease][approach][model] = best_stats["res_test"].get_dataframe_f1(col_name=f"{disease} {approach} $F_1$")

                            #if run_slot_eval:
                            #best_stats["res_test"].print_out()
                            #print(best_stats["res_test"].get_dataframe().to_latex(index=False, multicolumn_format="c", longtable=True))
                            best_res_evals[disease][approach][modeltype][nd][model] = best_stats["res_test"]#.get_dataframe_f1(col_name=f"{disease} {approach} $F_1$")

                            for column in df:
                                column_length = max(df[column].astype(str).map(len).max(), len(column))
                                col_idx = df.columns.get_loc(column)
                                writer.sheets[f"{approach}-{disease}-{model}-{modeltype}-{'dec' if len(nd) == 0 else nd}"].set_column(col_idx + 1, col_idx + 1,
                                                                                          column_length)

                        # with open(csv_file_path) as csv_file:
                        #     csv_dict = csv.DictReader(csv_file, delimiter=',')
                        #     for row in csv_dict:
                        #         print(row)
                        if len(row) > 0:
                            summarydf.loc[len(summarydf)] = row
                        print("### Eval summary")
                        print(summarydf.round(2).to_latex(index=False, multicolumn_format="c", float_format="%.2f"))  # , column_format="c"
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

    print("### Best params raw")
    for hp in best_hyperparams:
        pprint(hp)
    #print(best_hyperparams)
    with open(bestparams_path, "wb") as pres_pkl_file:
        pickle.dump(best_hyperparams, pres_pkl_file)
