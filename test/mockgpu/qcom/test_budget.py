"""The scheduler budget bounds finite work without changing shader loop counts."""
import numpy as np
import pytest
from test.mockgpu.qcom.emu import Instruction
from test.mockgpu.qcom.test_control import END, I, R, add, branch, jump, move, state


def finite_reduction():
  # Three increments, with one comparison/backedge per increment: 11 steps.
  compare = Instruction('cmps.u', {'COND': 0}, dict(DST=R(248), SRC1=R(0), SRC2=I(3)), 2 << 61)
  return (move(0, 0), add(0, 0, 1), compare, branch(-2), END)


def test_finite_loop_completes_at_exact_budget(monkeypatch):
  monkeypatch.setenv('MOCK_QCOM_MAX_STEPS', '11')
  group = state()
  group.run(finite_reduction())
  np.testing.assert_array_equal(group.registers[0][0], [3, 3, 3, 3])
  assert (group.pc == -1).all()


def test_finite_loop_reports_exhaustion_before_end(monkeypatch):
  monkeypatch.setenv('MOCK_QCOM_MAX_STEPS', '10')
  group = state()
  with pytest.raises(RuntimeError, match='instruction budget exhausted'):
    group.run(finite_reduction())
  assert (group.pc == 4).all()


def test_configured_budget_still_bounds_nonterminating_loop(monkeypatch):
  monkeypatch.setenv('MOCK_QCOM_MAX_STEPS', '7')
  group = state()
  with pytest.raises(RuntimeError, match='instruction budget exhausted'):
    group.run((add(0, 0, 1), jump(-1)))
  np.testing.assert_array_equal(group.registers[0][0], [4, 4, 4, 4])


@pytest.mark.parametrize('limit', ['0', '-1'])
def test_nonpositive_budget_rejects_before_execution(monkeypatch, limit):
  monkeypatch.setenv('MOCK_QCOM_MAX_STEPS', limit)
  group = state()
  with pytest.raises(ValueError, match='MOCK_QCOM_MAX_STEPS must be positive'):
    group.run((move(0, 99), END))
  np.testing.assert_array_equal(group.registers[0][0], [0, 0, 0, 0])


@pytest.mark.parametrize('limit, completed', [(2, False), (3, True)])
def test_barrier_release_consumes_one_scheduler_step(monkeypatch, limit, completed):
  monkeypatch.setenv('MOCK_QCOM_MAX_STEPS', str(limit))
  group = state()
  program = (Instruction('bar', {}, {}, 7 << 61), END)
  if completed:
    group.run(program)
    assert (group.pc == -1).all()
  else:
    with pytest.raises(RuntimeError, match='instruction budget exhausted'):
      group.run(program)
    assert (group.pc == 1).all()
  assert not group.waiting.any()


@pytest.mark.parametrize('limit, completed', [(3, False), (5, True)])
def test_budget_completion_requires_every_divergent_lane_to_end(monkeypatch, limit, completed):
  monkeypatch.setenv('MOCK_QCOM_MAX_STEPS', str(limit))
  group = state(lanes=2)
  group.registers[0][248] = [1, 0]
  program = (branch(3), move(0, 1), END, move(0, 2), END)
  if completed:
    group.run(program)
    np.testing.assert_array_equal(group.registers[0][0], [2, 1])
    assert (group.pc == -1).all()
  else:
    with pytest.raises(RuntimeError, match='instruction budget exhausted'):
      group.run(program)
    np.testing.assert_array_equal(group.pc, [3, -1])
    np.testing.assert_array_equal(group.registers[0][0], [0, 1])
