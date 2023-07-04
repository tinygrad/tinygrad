import os
import time
from tensorboard.compat.proto import event_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from extra.tensorboard.summary import histogram, scalar, image, hparams


class FileWriter:
  def __init__(self, log_dir, max_queue, flush_secs, filename_suffix):
    self.writer = EventFileWriter(str(log_dir), max_queue, flush_secs, filename_suffix)
  def add_event(self, event, step=None, walltime=None):
    event.wall_time = time.time() if walltime is None else walltime
    if step is not None: event.step = int(step)
    self.writer.add_event(event)
  def add_summary(self, summary, global_step=None, walltime=None):
    self.add_event(event_pb2.Event(summary=summary), global_step, walltime)
  def flush(self): self.writer.flush()
  def close(self): self.writer.close()
  def get_logdir(self): return self.writer.get_logdir()

class TinySummaryWriter:
  def __init__(self, log_dir=None, max_queue=10, flush_secs=120, filename_suffix=""):
    self.writer = FileWriter(log_dir, max_queue, flush_secs, filename_suffix)
  def __enter__(self): return self
  def __exit__(self, exc_type, exc_val, exc_tb): self.close()
  def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None):
    if type(hparam_dict) is not dict or type(metric_dict) is not dict:
      raise TypeError("hparam_dict and metric_dict should be dictionary.")
    exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)
    logdir = os.path.join(self.writer.get_logdir(), run_name if run_name else str(time.time()))
    with TinySummaryWriter(logdir) as w_hp:
      for summary in [exp, ssi, sei]: w_hp.writer.add_summary(summary)
      for k, v in metric_dict.items(): w_hp.add_scalar(k, v)
  def add_scalar(self, tag, value, global_step=None, walltime=None):
    self.writer.add_summary(scalar(tag, value), global_step, walltime)
  def add_histogram(self, name, values, bins, max_bins=None, global_step=None, walltime=None):
    self.writer.add_summary(histogram(name, values, bins, max_bins), global_step, walltime)
  def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats="CHW"):
    self.writer.add_summary(image(tag, img_tensor, dataformats=dataformats), global_step, walltime)
  def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats="NCHW"):
    self.add_image(tag, img_tensor, global_step, walltime, dataformats)
  def flush(self): self.writer.flush()
  def close(self): self.writer.close()
