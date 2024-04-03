from typing import List
from tinygrad.ops import ScheduleItem
from tinygrad.engine.commandqueue import CommandQueue

def run_schedule(schedule:List[ScheduleItem]): CommandQueue(schedule)()
