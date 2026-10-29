from voiceclone.train import TrainParams, checkpoint_dir


def test_each_checkpoint_gets_its_own_folder(tmp_path):
    # earlier adapters must survive later saves, and sort in training order
    dirs = [checkpoint_dir(tmp_path, s) for s in (50, 100, 1000)]
    assert len(set(dirs)) == 3
    assert sorted(d.name for d in dirs) == [d.name for d in dirs]
    assert all(d.parent == tmp_path for d in dirs)


def test_checkpoints_kept_by_default():
    assert TrainParams().save_every > 0
