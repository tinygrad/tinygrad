import time
from tensorboard.compat.proto import event_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from extra.tensorboard.summary import histogram, scalar, image

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

class TinySummaryWriter:
  def __init__(self, log_dir=None, max_queue=10, flush_secs=120, filename_suffix=""):
    self.writer = FileWriter(log_dir, max_queue, flush_secs, filename_suffix)
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