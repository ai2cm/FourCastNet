import json
import logging
import os

_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def config_logger(log_level=logging.INFO):
  logging.basicConfig(format=_format, level=log_level)

def log_to_file(logger_name=None, log_level=logging.INFO, log_filename='tensorflow.log'):

  if not os.path.exists(os.path.dirname(log_filename)):
    os.makedirs(os.path.dirname(log_filename))

  if logger_name is not None:
    log = logging.getLogger(logger_name)
  else:
    log = logging.getLogger()

  fh = logging.FileHandler(log_filename)
  fh.setLevel(log_level)
  fh.setFormatter(logging.Formatter(_format))
  log.addHandler(fh)

def log_versions():
  import torch
  import subprocess

  logging.info('--------------- Versions ---------------')
  logging.info('Torch: ' + str(torch.__version__))
  logging.info('----------------------------------------')


def write_to_metrics_file(key, value):
  """Creates or opens existing metrics.json and writes key value pair to it."""
  # hardcoded intentionally to match the beaker spec
  metrics_file = "/output/metrics.json"
  if os.path.exists(metrics_file):
    with open(metrics_file, "r+") as f:
        lines = "".join(f.readlines())
        metrics = json.loads(lines)
        metrics[key] = value
        f.seek(0)
        json.dump(metrics, f)
        f.truncate()
  else:
    with open(metrics_file, "w+") as f:
        json.dump({key: value}, f)

