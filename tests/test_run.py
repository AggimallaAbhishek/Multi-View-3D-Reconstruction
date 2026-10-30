from src.run import create_run_dir, is_run_dir_name


def test_run_dir_names_are_recognised(tmp_path):
    first = create_run_dir(tmp_path, "web_reconstruct")
    second = create_run_dir(tmp_path, "web_reconstruct")  # same second: suffixed
    assert is_run_dir_name(first.name, "web_reconstruct")
    assert is_run_dir_name(second.name, "web_reconstruct")
    assert not is_run_dir_name(first.name, "reconstruct")
    assert not is_run_dir_name(f"{first.name}_copy", "web_reconstruct")
    assert not is_run_dir_name("..", "web_reconstruct")
