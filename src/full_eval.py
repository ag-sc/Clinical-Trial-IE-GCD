import csv
import os
import pickle
from argparse import ArgumentParser
from datetime import datetime
import generative_approach.evaluation as gen

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--gen', required=False, default=None)
    argparser.add_argument('--gen-out',
                           required=False,
                           default=None)
    argparser.add_argument('--nodecoding', action='store_true')
    arguments = argparser.parse_args()

    gen_out = arguments.gen_out
    if gen_out is None:
        gen_out = arguments.gen

    gen_path = arguments.gen

    if gen_path is not None:
        gen_out_path = os.path.join(gen_out, datetime.today().strftime('%Y-%m-%d') + "-gen"+("-nodec" if arguments.nodecoding else "")+"-stats.csv")
        gen_pickle_path = os.path.join(gen_out, datetime.today().strftime('%Y-%m-%d') + "-gen"+("-nodec" if arguments.nodecoding else "")+"-stats.pkl")

        gen_stats = []
        extr_eval: csv.DictWriter

        with open(gen_out_path, "w", newline='') as gen_eval_file:
            fieldnames = ["model",
                          "validation-precision", "validation-recall", "validation-f1",
                          "test-precision", "test-recall", "test-f1"]

            gen_eval = csv.DictWriter(gen_eval_file, fieldnames=fieldnames)
            gen_eval.writeheader()

            gen_models = sorted([os.path.join(gen_path, name)
                                 for name in os.listdir(gen_path)
                                 if os.path.isdir(os.path.join(gen_path, name))])

            gen_configs = [os.path.join(gen_path, name)
                           for name in os.listdir(gen_path)
                           if os.path.isfile(os.path.join(gen_path, name))
                           and name.startswith("config_") and name.endswith(".json")]

            gen_tokenizers = sorted([os.path.join(gen_path, name)
                                     for name in os.listdir(gen_path)
                                     if os.path.isfile(os.path.join(gen_path, name))
                                     and name.startswith("tokenizer_gen_") and name.endswith(".json")])

            assert len(gen_configs) == 1
            gen_config = gen_configs[0]

            for model, tokenizer_path in zip(gen_models, gen_tokenizers):
                try:
                    print(model)
                    res_valid = gen.cli_eval(task_config=gen_config,
                                             model_path=model,
                                             tokenizer_path=tokenizer_path,
                                             use_validation=True,
                                             no_eval_decoding=arguments.nodecoding if arguments.nodecoding else None)
                    stats_valid = res_valid.compute_micro_stats()

                    res_test = gen.cli_eval(task_config=gen_config,
                                            model_path=model,
                                            tokenizer_path=tokenizer_path,
                                            use_validation=False,
                                            no_eval_decoding=arguments.nodecoding if arguments.nodecoding else None)
                    stats_test = res_test.compute_micro_stats()

                    print({
                        "model": model,
                        "validation-precision": stats_valid.precision(),
                        "validation-recall": stats_valid.recall(),
                        "validation-f1": stats_valid.f1(),
                        "test-precision": stats_test.precision(),
                        "test-recall": stats_test.recall(),
                        "test-f1": stats_test.f1()
                    }, flush=True)

                    gen_eval.writerow({
                        "model": model,
                        "validation-precision": stats_valid.precision(),
                        "validation-recall": stats_valid.recall(),
                        "validation-f1": stats_valid.f1(),
                        "test-precision": stats_test.precision(),
                        "test-recall": stats_test.recall(),
                        "test-f1": stats_test.f1()
                    })

                    gen_stats.append({
                        "model": model,
                        "res_valid": res_valid,
                        "res_test": res_test,
                    })
                except Exception as e:
                    print(e)
                    print("Failed to evaluate model: " + model)
                    continue

        with open(gen_pickle_path, "wb") as gen_pkl_file:
            pickle.dump(gen_stats, gen_pkl_file)
