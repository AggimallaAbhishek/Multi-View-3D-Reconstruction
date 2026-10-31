from src.run import create_run_dir, is_run_dir_name, run_stamp_of


def test_run_dir_names_are_recognised(tmp_path):
    first = create_run_dir(tmp_path, "web_reconstruct")
    second = create_run_dir(tmp_path, "web_reconstruct")  # same second: suffixed
    assert is_run_dir_name(first.name, "web_reconstruct")
    assert is_run_dir_name(second.name, "web_reconstruct")
    assert not is_run_dir_name(first.name, "reconstruct")
    assert not is_run_dir_name(f"{first.name}_copy", "web_reconstruct")
    assert not is_run_dir_name("..", "web_reconstruct")


def test_run_dir_can_be_given_its_stamp(tmp_path):
    first = create_run_dir(tmp_path, "web_reconstruct", stamp="20261031-132756")
    second = create_run_dir(tmp_path, "web_reconstruct", stamp="20261031-132756")
    assert (first.name, second.name) == (
        "20261031-132756_web_reconstruct",
        "20261031-132756_web_reconstruct_2",
    )
    assert run_stamp_of(second.name) == "20261031-132756"
    assert run_stamp_of("not_a_run") is None
