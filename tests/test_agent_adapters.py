from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.models.xraypro import ModelConfig, XRayProModel
from src.runtime.checkpoints import load_checkpoint
from tools.build_inference_bundle import build_bundle
from tools.import_legacy_checkpoint import convert


class AgentAdapterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = ModelConfig(class_space_groups=[1, 2, 4, 7, 14, 18, 19, 29, 33, 61, None])
        self.config_path = self.root / 'config.json'
        self.config_path.write_text(json.dumps(self.config.to_dict()))
        self.manifest = self.root / 'profiles.csv'
        self.embeddings = self.root / 'embeddings.npz'
        self.output = self.root / 'bundle.npz'
        for name, value in [('a', 10.), ('b', 20.)]:
            np.save(self.root / f'{name}.npy', np.full(3400, value, dtype=np.float32))
        self.write_manifest(['a', 'b'])
        self.write_embeddings(['b', 'a'])

    def write_manifest(self, ids):
        with self.manifest.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['sample_id', 'pxrd_path'])
            writer.writerows((key, f'{key}.npy') for key in ids)

    def write_embeddings(self, ids):
        np.savez(self.embeddings, sample_id=np.asarray(ids),
                 smiles_embedding=np.stack([np.full(768, 1. if key == 'a' else 2., dtype=np.float32) for key in ids]))

    def build(self):
        build_bundle(self.manifest, self.embeddings, self.config_path, self.output)

    def test_join_by_identity_and_preserve_existing_output(self):
        self.build()
        with np.load(self.output) as data:
            self.assertEqual(data['sample_id'].tolist(), ['a', 'b'])
            np.testing.assert_array_equal(data['smiles_embedding'][:, 0], [1., 2.])
            np.testing.assert_array_equal(data['pxrd'][:, 0], [10., 20.])
        previous = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(previous, self.output.read_bytes())

    def test_bad_ids_and_invalid_signals_fail_without_output(self):
        for ids in (['a', 'a'], ['a'], ['a', 'b', 'c'], ['a', '']):
            with self.subTest(ids=ids):
                self.write_embeddings(ids)
                with self.assertRaises(ValueError):
                    self.build()
                self.assertFalse(self.output.exists())
        self.write_embeddings(['a', 'b'])
        for profile in (np.zeros(3400), np.full(3400, np.nan), np.ones(3399)):
            np.save(self.root / 'a.npy', profile)
            with self.assertRaises(ValueError):
                self.build()
            self.assertFalse(self.output.exists())

    def test_legacy_import_preserves_tensors_and_source(self):
        model = XRayProModel(self.config)
        mapping = {'graph.atom_projection.': 'graph.atom_proj.',
                   'graph.convolutions.': 'graph.convs.', 'graph.normalizations.': 'graph.norms.'}
        state = {}
        for key, value in model.state_dict().items():
            for public, old in mapping.items():
                if key.startswith(public):
                    key = old + key[len(public):]
                    break
            state[key] = value
        args = dict(pxrd_encoder='seres', graph_encoder='simple', smiles_mode='embedding',
                    hidden_dim=128, projection_dim=64, graph_layers=2, dropout=.1)
        source = self.root / 'original.pt'
        torch.save(dict(model_state=state, args=args, epoch=47), source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        output = self.root / 'converted.pt'
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            convert(source, self.config_path, output, '0' * 64, 'test fixture')
        bad_config = self.config.to_dict()
        bad_config['pxrd_length'] = 3300
        self.config_path.write_text(json.dumps(bad_config))
        with self.assertRaisesRegex(ValueError, '3400'):
            convert(source, self.config_path, output, digest, 'test fixture')
        self.config_path.write_text(json.dumps(self.config.to_dict()))
        convert(source, self.config_path, output, digest, 'synthetic test fixture only')
        loaded, _ = load_checkpoint(output, device=torch.device('cpu'))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
        with self.assertRaises(FileExistsError):
            convert(source, self.config_path, output, digest, 'test fixture')


if __name__ == '__main__':
    unittest.main()
