from cocofest.simulation.cpu_affinity import parse_cpu_list, reserved_cpu_ids


def test_linux_cpu_list_parser_and_dedicated_process_discovery(tmp_path):
    assert parse_cpu_list("0-2, 7, 9-10") == frozenset({0, 1, 2, 7, 9, 10})
    (tmp_path / "42").mkdir()
    (tmp_path / "42" / "status").write_text("Name:\tsolver\nCpus_allowed_list:\t30\n", encoding="utf-8")
    (tmp_path / "43").mkdir()
    (tmp_path / "43" / "status").write_text("Cpus_allowed_list:\t31-32\n", encoding="utf-8")
    assert reserved_cpu_ids(tmp_path, current_pid=-1) == frozenset({30})
