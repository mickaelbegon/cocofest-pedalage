"""Closing an active run must require explicit consent before stopping it."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cocofest.simulation.gui import SimulationApp


@pytest.mark.parametrize("confirmed", (False, True))
def test_closing_running_simulation_can_be_cancelled(monkeypatch, confirmed):
    prompt = Mock(return_value=confirmed)
    monkeypatch.setattr("tkinter.messagebox.askyesno", prompt)
    app = SimpleNamespace(_closing=False, process=SimpleNamespace(running=True),
                          window=Mock(), stop=Mock())

    SimulationApp.close(app)

    prompt.assert_called_once()
    assert app._closing is confirmed
    assert app.stop.call_count == int(confirmed)
    app.window.destroy.assert_not_called()


def test_closing_inactive_window_requires_no_confirmation(monkeypatch):
    prompt = Mock()
    monkeypatch.setattr("tkinter.messagebox.askyesno", prompt)
    app = SimpleNamespace(_closing=False, process=SimpleNamespace(running=False),
                          window=Mock(), stop=Mock())

    SimulationApp.close(app)

    prompt.assert_not_called()
    app.stop.assert_not_called()
    app.window.destroy.assert_called_once()
    assert app._closing


def test_second_close_does_not_interrupt_shutdown_again(monkeypatch):
    prompt = Mock()
    monkeypatch.setattr("tkinter.messagebox.askyesno", prompt)
    app = SimpleNamespace(_closing=True, process=SimpleNamespace(running=True),
                          window=Mock(), stop=Mock())

    SimulationApp.close(app)

    prompt.assert_not_called()
    app.stop.assert_not_called()
    app.window.destroy.assert_not_called()
