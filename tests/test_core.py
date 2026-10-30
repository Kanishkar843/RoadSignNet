import csv
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from roadsignnet.data import assert_disjoint, discover_test, discover_training, split_training
from roadsignnet.labels import CLASS_NAMES
from roadsignnet.predict import append_record, rank_scores
from roadsignnet.preprocess import preprocess_image


def image_bytes(mode='RGB'):
    out = BytesIO()
    Image.new(mode, (48, 24), 128 if mode == 'L' else (255, 0, 0)).save(out, format='PNG')
    return out.getvalue()


def test_class_mapping():
    assert len(CLASS_NAMES) == len(set(CLASS_NAMES)) == 43
    assert CLASS_NAMES[14] == 'Stop'
    assert CLASS_NAMES[33] == 'Turn right ahead'
    assert CLASS_NAMES[34] == 'Turn left ahead'


@pytest.mark.parametrize('mode', ['RGB', 'L'])
def test_preprocessing(mode):
    result = preprocess_image(image_bytes(mode))
    assert result.shape == (32, 32, 3)
    assert result.dtype == np.float32
    assert 0 <= result.min() <= result.max() <= 1
    if mode == 'RGB':
        np.testing.assert_array_equal(result[0, 0], [1, 0, 0])


@pytest.mark.parametrize('payload', [b'', b'not an image'])
def test_bad_image(payload):
    with pytest.raises(ValueError):
        preprocess_image(payload)


def test_gif_rejected():
    output = BytesIO()
    Image.new('RGB', (8, 8)).save(output, format='GIF')
    with pytest.raises(ValueError, match='JPEG'):
        preprocess_image(output.getvalue())


def test_track_split_and_duplicates():
    rows = [{'class_id': c, 'group': f'{c}:{g}', 'sha256': f'{c}-{g}-{f}', 'path': 'x'}
            for c in range(43) for g in range(5) for f in range(2)]
    train, val = split_training(rows)
    assert len(train) == 344 and len(val) == 86
    assert not {r['group'] for r in train} & {r['group'] for r in val}
    assert {r['class_id'] for r in train} == {r['class_id'] for r in val} == set(range(43))
    assert split_training(rows) == (train, val)
    with pytest.raises(ValueError, match='Identical'):
        assert_disjoint(train, [{**val[0], 'sha256': train[0]['sha256']}])
    with pytest.raises(ValueError, match='tracks overlap'):
        assert_disjoint(train, [{**val[0], 'group': train[0]['group']}])


def test_insufficient_tracks():
    with pytest.raises(ValueError, match='two distinct'):
        split_training([{'class_id': 0, 'group': '0:0', 'sha256': 'x'}])


def test_missing_dataset(tmp_path):
    with pytest.raises(ValueError, match='No GTSRB'):
        discover_training(tmp_path)


def test_test_csv_validation(tmp_path):
    csv_path = tmp_path / 'test.csv'
    csv_path.write_text('Path,ClassId\n../outside.png,0\n')
    with pytest.raises(ValueError, match='relative'):
        discover_test(tmp_path, csv_path)
    csv_path.write_text('Filename;ClassId\nsign.png;14\n')
    (tmp_path / 'sign.png').write_bytes(image_bytes())
    assert discover_test(tmp_path, csv_path)[0]['class_id'] == 14
    csv_path.write_text('Filename;ClassId\nsign.png;14.5\n')
    with pytest.raises(ValueError, match='integers'):
        discover_test(tmp_path, csv_path)


def test_scores_and_log(tmp_path):
    scores = np.zeros(43)
    scores[14], scores[1], scores[2] = 0.6, 0.3, 0.1
    result = rank_scores(scores)
    assert result['class_id'] == 14
    assert [r['class_id'] for r in result['top_three']] == [14, 1, 2]
    assert result['review_status'].startswith('Low confidence')
    result['timestamp'] = '2026-10-30T00:00:00+00:00'
    target = tmp_path / 'predictions.csv'
    append_record(target, '=formula.png', result)
    append_record(target, 'sign.png', result)
    with target.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and rows[0]['image_name'].startswith("'")


@pytest.mark.parametrize('scores', [np.zeros(43), np.ones(42)/42, np.full(43, np.nan)])
def test_invalid_model_scores(scores):
    with pytest.raises(ValueError):
        rank_scores(scores)
