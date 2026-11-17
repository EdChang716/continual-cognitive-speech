"""Private report content must never become executable HTML or JavaScript."""
from copy import deepcopy
from html.parser import HTMLParser
import json

import pytest

from ccspeech.data.transcript_report import render


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.attributes = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attributes.extend(attrs)


def fixture_report(tmp_path):
    payload = '</script><img src="https://invalid.example/steal" onerror="alert(1)"> & "text"'
    metric = {'metric': 'WER', 'S': 1, 'D': 1, 'I': 1, 'reference_units': 4,
              'hypothesis_units': 4, 'error_rate': .75}
    alignment = [
        {'tag': 'equal', 'reference_tokens': ['a'], 'hypothesis_tokens': ['a']},
        {'tag': 'replace', 'reference_tokens': [payload], 'hypothesis_tokens': ['replacement']},
        {'tag': 'delete', 'reference_tokens': ['deleted'], 'hypothesis_tokens': []},
        {'tag': 'insert', 'reference_tokens': [], 'hypothesis_tokens': [payload]},
    ]
    row = {'recording_id': payload, 'participant_id': 'synthetic-person', 'language': 'en',
           'split': 'train', 'reference_scope_verified': True, 'source_dataset': payload,
           'reference_scope': 'full', 'reference': payload, 'raw_asr': payload,
           'candidate_asr': payload, 'window_asr': payload,
           'counts': {name: deepcopy(metric) for name in ('raw', 'candidate', 'window')},
           'alignments': {name: deepcopy(alignment) for name in ('raw', 'candidate', 'window')},
           'quality_flags': [payload], 'source_word_count': 4, 'window_covered_word_count': 3,
           'retained_windows': 1}
    summary = {'schema_version': 1, 'recordings': 1, 'participants': 1,
               'groups': [{'language': 'en', 'split': 'train', 'reference_scope_verified': True,
                           'recordings': 1, 'participants': 1, 'reference_units': 4,
                           'conditions': {name: deepcopy(metric) for name in ('raw', 'candidate', 'window')}}],
               'notes': [payload]}
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    (tmp_path / 'differences.jsonl').write_text(json.dumps(row) + '\n')
    return payload, row, summary


def test_every_transcript_and_attribute_is_escaped(tmp_path):
    payload, _, _ = fixture_report(tmp_path)
    destination = render(tmp_path)
    text = destination.read_text()
    assert payload not in text
    assert '&lt;/script&gt;&lt;img' in text
    parser = Elements()
    parser.feed(text)
    assert parser.tags.count('script') == 1
    assert 'img' not in parser.tags
    assert not any(key.startswith('on') or key in {'src', 'href'} for key, _ in parser.attributes)
    assert '75.00%' in text and 'S 1 / D 1 / I 1' in text
    assert 'data-recording' in text
    assert not destination.with_suffix('.html.part').exists()


def test_duplicate_or_missing_recording_is_rejected(tmp_path):
    _, row, summary = fixture_report(tmp_path)
    summary['recordings'] = 2
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='coverage'):
        render(tmp_path)
    (tmp_path / 'differences.jsonl').write_text((json.dumps(row) + '\n') * 2)
    with pytest.raises(ValueError, match='duplicates'):
        render(tmp_path)


def test_unknown_alignment_and_symlink_output_are_rejected(tmp_path):
    _, row, _ = fixture_report(tmp_path)
    row['alignments']['raw'][0]['tag'] = 'unsupported'
    (tmp_path / 'differences.jsonl').write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='alignment'):
        render(tmp_path)
    fixture_report(tmp_path)
    other = tmp_path / 'existing.txt'
    other.write_text('preserve')
    (tmp_path / 'review.html').symlink_to(other)
    with pytest.raises(ValueError, match='symlink'):
        render(tmp_path)
    assert other.read_text() == 'preserve'
