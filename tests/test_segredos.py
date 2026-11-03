"""The local .env reader (``core.segredos``): quotes, inline comments, the last value wins, the environment first."""

from core.segredos import ler_env, segredo


def test_quoted_values_keep_spaces_inline_comments_drop_and_the_last_value_wins(tmp_path):
    arquivo = tmp_path / ".env"
    arquivo.write_text('# comment\nCHAVE=primeira\nCHAVE=segunda\nSENHA="abcd efgh ijkl mnop" # Senha de app\n'
                       "SOLTA=valor # nota\nVAZIA=\nexport OUTRA='x y'\n", encoding="utf-8")
    assert ler_env(arquivo) == {"CHAVE": "segunda", "SENHA": "abcd efgh ijkl mnop", "SOLTA": "valor", "VAZIA": "",
                                "OUTRA": "x y"}
    assert ler_env(tmp_path / "nao_existe") == {}


def test_the_file_beats_a_stale_environment_variable_and_an_empty_value_is_none(tmp_path, monkeypatch):
    arquivo = tmp_path / ".env"
    arquivo.write_text("A=do_arquivo\nB=\n", encoding="utf-8")
    monkeypatch.setenv("A", "do_ambiente")
    monkeypatch.setenv("D", "so_no_ambiente")
    monkeypatch.delenv("B", raising=False)
    assert segredo("A", arquivo) == "do_arquivo"
    assert segredo("D", arquivo) == "so_no_ambiente"
    assert segredo("B", arquivo) is None and segredo("C", arquivo) is None
