import os
from random import random, randint

import numpy as np
import pandas as pd
import mlflow
import json
from mlflow import log_metric, log_param, log_params, log_artifacts
from hangar import Repository

if __name__ == "__main__":
  # #  Log a parameter (key-value pair)
  #   mlflow.create_experiment('a')
  #   mlflow.create_experiment('b')
  #   mlflow.set_experiment("a")
  #   #mlflow.set_experiment("b")

    mlflow.start_run()
    #log_params({"param1": randint(0, 100), "param2": randint(0, 100), 'exp': ['a', 'b', 'c'], 'try':4})


  #   log_param("config_value", randint(0, 100))
  #
  #   # Log a dictionary of parameters
  #   log_params({"param1": randint(0, 100), "param2": randint(0, 100), 'exp':['a', 'b', 'c']})
  #
  #   # Log a metric; metrics can be updated throughout the run
  #   log_metric("accuracy", random() / 2.0)
  #   log_metric("accuracy", random() + 0.1)
  #   log_metric("accuracy", random() + 0.2)
  #
    # Log an artifact (output file)
    # if not os.path.exists("outputs"):
    #     os.makedirs("outputs")
    # with open("outputs/test.txt", "w") as f:
    #     f.write("hello world!")
    # log_artifacts("outputs")
    #
    # mlflow.end_run()
    # run = mlflow.search_runs(output_format="list")[0]  # [0]
    # print(run.data.art)
  #   mlflow.end_run()
  #   print(mlflow.last_active_run())
  #   run = mlflow.search_runs(experiment_names=['a'], output_format="list")[0]
  #   print(run.info.run_id)
    # runs = mlflow.search_runs(
    #     output_format="list"
    # )
    # last_run = runs[-1]
    # print(last_run.data.params)
    #run = mlflow.search_runs( 2855f1477a9144ddb775738b7cf29865

    ########################################

    repo = Repository(path="C:\\Users\\slok\\PycharmProjects\\data-quality-dashboard\\test_repo")
    # if not repo.initialized:
    #repo.init(user_name="Alessia Marcolini", user_email="alessia@tensorwerk.com")
    co = repo.checkout(write=True)

    data = {'col_1': [0, 0, 0, 1], 'col_2': ['a', 'b', 'c', 'd']}
    df = pd.DataFrame.from_dict(data)

    dummy = np.random.rand(3, 2)
    dummy_col = co.add_ndarray_column(name="dummy_column", prototype=dummy)
    dummy_col[0] = dummy
    #data_col = co.add_str_column(name='data')
    #data_col[0] = df.to_json()
    co.commit('hello world, this is my second hangar commit')
    print(co.log())

    co.close()
    repo.summary()

