"""Synthetic integration only: proves software execution, not sign recognition accuracy."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from roadsignnet.artifacts import load_bundle
from roadsignnet.data import discover_training
from roadsignnet.evaluate import evaluate
from roadsignnet.predict import predict_image
from roadsignnet.train import train


def make_fixture(root):
    rng = np.random.default_rng(2026)
    for class_id in range(43):
        folder = root / 'Train' / str(class_id)
        folder.mkdir(parents=True)
        for track in range(3):
            for frame in range(2):
                pixels = rng.integers(0, 256, (34, 38, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f'{class_id:05d}_{track:05d}_{frame:05d}.png')
    (root / 'Test').mkdir()
    rows = []
    for class_id in range(43):
        name = f'Test/{class_id:05d}.png'
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(root / name)
        rows.append({'Path': name, 'ClassId': class_id})
    pd.DataFrame(rows).to_csv(root / 'Test.csv', index=False)


def test_cpu_training_reload_prediction_and_evaluation(tmp_path, monkeypatch):
    root, bundle, results = tmp_path / 'data', tmp_path / 'bundle', tmp_path / 'results'
    make_fixture(root)
    metadata = train(root, bundle, epochs=1, batch_size=32, augment=True)
    assert metadata['train_samples'] == 172
    assert metadata['validation_samples'] == 86
    assert not any('GPU' in device for device in metadata['devices'])
    model, _ = load_bundle(bundle)
    image = root / 'Test/00000.png'
    result = predict_image(model, image)
    assert len(result['top_three']) == 3 and 0 <= result['score'] <= 1
    reloaded, _ = load_bundle(bundle)
    result2 = predict_image(reloaded, image)
    assert result['class_id'] == result2['class_id']
    assert result['score'] == pytest.approx(result2['score'], abs=1e-7)
    # Exercise the actual app's prediction and invalid-image rendering with an
    # in-memory upload; AppTest does not yet expose a native file-upload setter.
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('ROADSIGNNET_MODEL_DIR', str(bundle))
    app_path = str(Path(__file__).resolve().parents[1] / 'app.py')
    upload = SimpleNamespace(name='synthetic-test.png', getvalue=image.read_bytes)
    with patch('streamlit.file_uploader', return_value=upload):
        app = AppTest.from_file(app_path).run(timeout=30)
        assert not app.exception and not app.error
        assert app.metric[0].label == 'Top model score'
        assert len(app.get('download_button')) == 1
    invalid = SimpleNamespace(name='broken.png', getvalue=lambda: b'broken')
    with patch('streamlit.file_uploader', return_value=invalid):
        app = AppTest.from_file(app_path).run(timeout=30)
        assert not app.exception
        assert 'Cannot decode' in app.error[0].value
    metrics = evaluate(root, root / 'Test.csv', bundle, results, batch_size=16)
    assert metrics['test_samples'] == 43
    assert 0 <= metrics['accuracy'] <= 1
    assert (results / 'confusion_matrix.png').stat().st_size > 100
    matrix = pd.read_csv(results / 'confusion_matrix.csv', index_col=0)
    assert matrix.shape == (43, 43) and matrix.to_numpy().sum() == 43
    # A test annotation pointing at training images must be rejected.
    training_image = discover_training(root)[0]['path']
    pd.DataFrame([{'Path': str(Path(training_image).relative_to(root)), 'ClassId': 0}]).to_csv(root / 'leaky.csv', index=False)
    with pytest.raises(ValueError, match='Identical'):
        evaluate(root, root / 'leaky.csv', bundle, tmp_path / 'bad-results')
    original = json.loads((bundle / 'metadata.json').read_text())
    broken = {**original, 'class_names': list(reversed(original['class_names']))}
    (bundle / 'metadata.json').write_text(json.dumps(broken))
    with pytest.raises(ValueError, match='class ordering'):
        load_bundle(bundle)
    (bundle / 'metadata.json').write_text(json.dumps(original))
    with (bundle / 'model.keras').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        load_bundle(bundle)


def test_official_layout(tmp_path):
    rng = np.random.default_rng(12)
    for class_id in range(43):
        folder = tmp_path / 'GTSRB/Final_Training/Images' / f'{class_id:05d}'
        folder.mkdir(parents=True)
        Image.fromarray(rng.integers(0, 256, (12, 12, 3), dtype=np.uint8)).save(folder / '00001_00000.ppm')
    records = discover_training(tmp_path)
    assert len(records) == 43
    assert records[42]['group'] == '42:00001'


def test_app_without_model(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from pathlib import Path
    monkeypatch.setenv('ROADSIGNNET_MODEL_DIR', str(tmp_path / 'missing'))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == 'RoadSignNet'
    assert any('Train RoadSignNet' in item.value for item in app.info)
