"""Serialize exact real scalar inputs; no local model fit or inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import gc
import hashlib
import shutil
import sys
import zipfile
import numpy as np
from common112 import *
os.environ.update(ENV_PINS)

sys.path.insert(0, str(HERE.parent / 'b106_lstm'))
import common106 as C
sys.path.insert(0, str(HERE.parent / 'b75_availability'))
import stage_fits as SF
sys.path.insert(0, str(HERE.parent / 'b78_sparse'))
import common78


def ah(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    guard()
    C.guard()
    C.inputs_guard()
    prior = ROOT / 'data/colab_b111'
    audited = read(ROOT / 'docs/paper/b111_comparison.json')
    assert audited['independent_audit']['all_pass']
    assert read(prior / 'independent_audit.json')['all_pass']
    assert sha(prior / 'b111_results.zip') == audited['independent_audit']['archive_sha256']
    assert not OUT.exists(), 'Preserve the previous export/attempt; no overwrite'
    OUT.mkdir()
    bundle = OUT / 'bundle'
    bundle.mkdir()

    old_private = read(prior / 'private_map.json')
    old_manifest = read(prior / 'bundle/manifest.json')
    old_result_hashes = read(prior / 'retrieved/result_hashes.json')
    codes = np.asarray(old_private['row_codes'], dtype=np.int64)
    back = np.asarray(old_private['row_inverse'], dtype=np.int64)
    assert np.array_equal(codes[back], np.arange(len(back)))
    file_codes = {int(k): int(v) for k, v in old_private['file_codes'].items()}
    old_rows = npz(prior / 'bundle/rows.npz')
    ctx = SF.Ctx()
    R = ctx.R
    assert len(codes) == len(R['gi'])
    assert np.array_equal(old_rows['truth'], R['tg_F'][back], equal_nan=True)
    assert np.array_equal(old_rows['discharge'][codes],
                          np.array([file_codes[int(f)] for f in R['fidx']]))
    assert len(set(codes.tolist())) == len(codes)

    reference_hashes = {}
    def bind(path):
        path = Path(path)
        reference_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    for path in (prior / 'private_map.json', prior / 'bundle/rows.npz',
                 prior / 'bundle/manifest.json', prior / 'independent_audit.json',
                 ROOT / 'docs/paper/b111_comparison.json',
                 ROOT / 'data/.b106/provenance.json',
                 ROOT / 'data/.b105/provenance.json',
                 SF.K.C.CACHE / 'rows.npz'):
        bind(path)
    for rule in RULES:
        base = ROOT / 'data' / ('.b78' if rule == 'K10' else '.b75')
        bind(base / 'inputs' / (rule + '.npz'))
        bind(base / 'sets.npz')

    units = {}
    private = {'mapping_source_sha256': sha(prior / 'private_map.json'),
               'units': {}}
    total_rows = 0
    bit_checks = 0
    for ui, name in enumerate(UNITS):
        tag = 'U' + str(ui)
        prep = C.prepared(name + '__outer')
        man = prep['identity']['manifest']
        stats = prep['identity']['stats']
        assert stats['train_files'] == sorted(man['train'])
        assert not man['held']
        sets_files = {split: set(man[split]) for split in ('train', 'val', 'test')}
        assert all(not sets_files[a] & sets_files[b]
                   for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')))
        if name.startswith('V2'):
            groups = {k: set(R['clus'][np.isin(R['fidx'], v)].tolist())
                      for k, v in sets_files.items()}
            assert all(not groups[a] & groups[b]
                       for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')))
        bind(ROOT / 'data/.b106/prepared' / (name + '__outer.pt'))
        bind(ROOT / 'data/.b106/prepared' / (name + '__outer.json'))
        entry = {'name': name,
                 'files': {k: [file_codes[int(f)] for f in man[k]]
                           for k in ('train', 'val', 'test')},
                 'normalization_files_sha256': C.digest(sorted(man['train'])),
                 'manifest_sha256': C.digest(man),
                 'original_stats_sha256': C.digest(stats),
                 'split_hashes': {}}
        private['units'][tag] = {'manifest': man, 'stats': stats, 'rows': {}}
        del prep
        gc.collect()
        fast, mb, me = ctx.unit_arrays({'fit': man['train']})
        # Verify the same TRAIN-only fast summaries used by the audited B111.
        assert np.array_equal(fast[back], npz(prior / 'bundle' / (tag + '_fast.npz'))['fast'],
                              equal_nan=True)
        masks = {k: np.isin(R['fidx'], v) for k, v in sets_files.items()}
        for rule in RULES:
            if rule == 'K10':
                I = common78.load_inputs(rule)
                H = SF.K.history_matrix(R['tstep'], I)
                A = SF.K.anchor_scalars(R['gi'], I['anc_g'], mb, me,
                                        ctx.mr_b, ctx.mr_e)
                XS = SF.Lazy(H, fast, A)
                gp = I['G']
                score_sets = npz(ROOT / 'data/.b78/sets.npz')
            else:
                XS, gp = ctx.matrices(rule, fast, mb, me)
                score_sets = ctx.S
            scored = score_sets[('S_' if rule == 'S' else 'common_') + 'F']
            split_rows = {}
            parts = []
            hashes = {}
            for split, number in (('train', 0), ('val', 1), ('test', 2)):
                if rule == 'K10' and split != 'test':
                    continue
                ii = np.flatnonzero(scored & masks[split])
                ii = ii[np.argsort(codes[ii])]
                assert len(ii) >= 30
                assert len(np.unique(R['fidx'][ii])) >= 2
                assert len(np.unique(R['clus'][ii])) >= 2
                x = np.asarray(XS['TX'][ii], dtype=np.float64)
                # A separate concatenation path checks the helper's column order
                # without fitting, floating conversion or row filtering.
                manual = np.column_stack((XS.H10[ii], XS.f8[ii], XS.A[ii]))
                assert np.array_equal(x, manual)
                y = np.asarray(R['tg_F'][ii], dtype=np.float64)
                g = np.asarray(gp[ii], dtype=np.float64)
                assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(g).all()
                assert x.ndim == 2 and len(x) == len(y)
                entry.setdefault('features', int(x.shape[1]))
                assert entry['features'] == x.shape[1]
                z = {'x': x, 'y': y, 'gp': g, 'idx': codes[ii],
                     'split': np.full(len(ii), number, dtype=np.int8),
                     'cluster': old_rows['cluster'][codes[ii]],
                     'discharge': old_rows['discharge'][codes[ii]]}
                assert np.all(np.diff(z['idx']) > 0)
                hashes[split] = {'rows': len(ii),
                                 'files': len(np.unique(z['discharge'])),
                                 'clusters': len(np.unique(z['cluster'])),
                                 **{k + '_sha256': ah(z[k]) for k in ('x', 'y', 'gp', 'idx')},
                                 'original_rows_sha256': ah(ii)}
                parts.append(z)
                split_rows[split] = ii.tolist()
                total_rows += len(ii)
                bit_checks += 1
            all_rows = {k: np.concatenate([p[k] for p in parts], axis=0)
                        for k in parts[0]}
            assert len(np.unique(all_rows['idx'])) == len(all_rows['idx'])
            # Column-major storage improves compression without changing any value.
            all_rows['x'] = np.asfortranarray(all_rows['x'])
            np.savez_compressed(bundle / (tag + '_' + rule + '.npz'), **all_rows)
            entry['split_hashes'][rule] = hashes
            private['units'][tag]['rows'][rule] = split_rows

            ref_path = prior / 'bundle' / (tag + '_' + rule + '_reference.npz')
            estimate_path = prior / 'retrieved/estimates' / (tag + '_' + rule + '.npz')
            assert sha(ref_path) == old_manifest['files'][ref_path.name]
            assert sha(estimate_path) == old_result_hashes['estimates/' + estimate_path.name]
            bind(ref_path)
            bind(estimate_path)
            ref = npz(ref_path)
            est = npz(estimate_path)
            test = parts[-1]
            assert np.array_equal(test['split'], np.full(len(test['idx']), 2, dtype=np.int8))
            assert np.array_equal(ref['idx'], test['idx'])
            assert np.array_equal(est['idx'], ref['idx'])
            assert np.array_equal(ref['truth'], test['y'])
            assert np.array_equal(ref['GP_only'], test['gp'])
            for key in ref:
                assert np.array_equal(est[key], ref[key])
            for key in ('LSTM', 'LSTM_frozen_D10', 'LSTM_mixed_post'):
                ref[key] = est[key]
            assert all(np.isfinite(v).all() for v in ref.values())
            np.savez_compressed(bundle / (tag + '_' + rule + '_reference.npz'), **ref)
            print('B112 exported ' + tag + ' ' + rule + ' rows=' + str(len(all_rows['idx'])), flush=True)
            del all_rows, parts
        units[tag] = entry
        del fast, mb, me
        gc.collect()

    for filename in ('common112.py', 'model112.py', 'run112.py', 'stats112.py'):
        shutil.copyfile(HERE / filename, bundle / filename)
    manifest = {'registration': 'PREREGISTRATION_B112.md', 'sources': sources(),
                'units': units, 'reference_sha256': reference_hashes,
                'mapping_source_sha256': private['mapping_source_sha256'],
                'anonymized': True, 'TEST_exported': True,
                'target': 'current_report_F', 'new_local_fits': 0,
                'all_inputs_float64': True, 'input_helper_bit_equal_checks': bit_checks,
                'inherited_input_provenance_sha256': C.digest(C.prov()),
                'files': {p.name: sha(p) for p in sorted(bundle.iterdir())}}
    write(bundle / 'manifest.json', manifest)
    write(OUT / 'private_map.json', private)
    archive = OUT / 'ces_b112_bundle.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(bundle.iterdir()):
            z.write(path, path.name)
    write(OUT / 'export_verification.json', {'all_pass': True,
          'archive_sha256': sha(archive), 'bytes': archive.stat().st_size,
          'sources': sources(), 'total_exported_rows': total_rows,
          'input_helper_bit_equal_checks': bit_checks, 'TEST_exported': True,
          'local_fits': 0, 'local_GPU_work': 0})
    print('B112 bundle bytes=' + str(archive.stat().st_size), flush=True)


if __name__ == '__main__':
    main()
