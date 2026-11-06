"""The lake command the paper routines run on their due days (DECISOES 285): the lake's base-Python runner when it
exists (the venv's python.exe cannot write outside the project since 2026-11-02), else the venv's python as before."""


def test_with_the_base_runner_the_lake_command_goes_through_it(tmp_path):
    from core.python_do_lago import comando_do_lago
    local, roaming = tmp_path / "local", tmp_path / "roaming"
    runner = local / "b3lake" / "run_base.py"
    python = roaming / "uv" / "python" / "cpython-3.14-windows-x86_64-none" / "python.exe"
    for arquivo in (runner, python):
        arquivo.parent.mkdir(parents=True)
        arquivo.write_text("")
    assert comando_do_lago(tmp_path / "lago", "universe", local=local, roaming=roaming) == [
        str(python), str(runner), "b3lake.cli", "universe"]


def test_without_the_runner_it_falls_back_to_the_lakes_venv_python(tmp_path):
    from core.python_do_lago import comando_do_lago
    assert comando_do_lago(tmp_path, "fundamentals", local=tmp_path / "x", roaming=tmp_path / "y") == [
        str(tmp_path / ".venv" / "Scripts" / "python.exe"), "-m", "b3lake.cli", "fundamentals"]
