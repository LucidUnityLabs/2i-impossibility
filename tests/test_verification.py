"""Exact positive controls, negative controls, and CLI/reproducibility regressions."""
from contextlib import redirect_stdout, redirect_stderr
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from exact_2i import group, Zphi, qmul, VerificationError, require
from exact_sl2 import character_data
import supplement_verify as supplement
import verify_2I_reps as reps
import wilson_line_centralizer as wilson
import gamma70_fusion as gamma
import eclectic_nonsplit as eclectic
import compactification_search as compact
from verification_io import encode,atomic_write,cli


class ExactGroupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = group()

    def test_ring_relation(self):
        phi = Zphi(0,1)
        self.assertEqual(phi*phi,phi+1)
        self.assertEqual(phi.conjugate_field(),1-phi)

    def test_reject_float_coefficients(self):
        with self.assertRaises(VerificationError):
            Zphi(1.0,0)

    def test_reject_nonintegral_quotient(self):
        with self.assertRaises(VerificationError):
            Zphi(1).div(2)

    def test_group_order_and_exponent(self):
        self.assertEqual((self.g.n,self.g.exponent),(120,60))

    def test_all_associativity_triples(self):
        T = self.g.table
        # 120^3 exact table identities, independently checking the encoded algebra.
        self.assertTrue(all(T[T[a][b]][c] == T[a][T[b][c]]
                            for a in range(120) for b in range(120) for c in range(120)))

    def test_class_sizes(self):
        self.assertEqual(sorted(self.g.sizes),[1,1,12,12,12,12,20,20,30])

    def test_character_orthogonality(self):
        for a,ca in self.g.chars.items():
            for b,cb in self.g.chars.items():
                self.assertEqual(self.g.inner(ca,cb),int(a == b))

    def test_doublet_tensor_has_invariant(self):
        c = self.g.chars['2a']
        self.assertEqual(self.g.decompose(self.g.mul(c,c)),{'1':1,'3a':1})

    def test_fs_and_faithfulness(self):
        self.assertEqual([k for k in self.g.chars if self.g.faithful[k]],['2a','2b','4H','6'])
        self.assertEqual(sum(self.g.fs[k]*d for k,d in self.g.dims.items()),2)

    def test_original_gamma_fs_list_fails_frobenius_count(self):
        old = [1,-1,-1,1,1,-1,-1,1,1]
        dims = [1,2,2,3,3,4,4,5,6]
        self.assertEqual(sum(a*b for a,b in zip(old,dims)),6)
        self.assertNotEqual(6,2)

    def test_character_corruption_rejected(self):
        v = list(self.g.chars['1'])
        v[self.g.id_class] += 1
        with self.assertRaises(VerificationError):
            self.g.decompose(tuple(v))

    def test_negative_multiplicity_rejected(self):
        with self.assertRaises(VerificationError):
            self.g.decompose(self.g.scale(-1,self.g.chars['1']))

    def test_highest_weight_validation(self):
        for n in (-1,0.5,True):
            with self.assertRaises(VerificationError):
                self.g.su2(n)

    def test_molien_and_first_positive_spin(self):
        self.assertEqual(supplement.molien(self.g)[:13],[1]+[0]*11+[1])
        self.assertEqual([self.g.invariant_spin(2*j) for j in range(7)],[1,0,0,0,0,0,1])


class DerivedTests(unittest.TestCase):
    def test_cubic_arithmetic(self):
        a = supplement.arithmetic()
        self.assertEqual(a['degree'],3)
        self.assertFalse(a['algebraic_integer'])
        self.assertFalse(a['contained_in_Q_zeta60'])

    def test_spinor_exhaustiveness_and_unique_identity(self):
        r = reps.spinor_search(group())
        self.assertEqual(r['total_real_eight_partitions'],25)
        self.assertEqual(len(r['faithful_partitions']),12)
        self.assertEqual(r['unique_three_singlet_piece'],{'4H':1})
        for c in r['faithful_partitions']:
            for s in c['half_spinors']:
                self.assertEqual(sum(group().dims[k]*m for k,m in s.items()),8)

    def test_spinor_branching_is_derived_from_weights(self):
        s,t = reps.spinor_characters_su2(group(),(3,3,1,1))
        self.assertEqual(group().decompose(s),{'1':3,'5':1})
        self.assertEqual(group().decompose(t),{'4H':2})
        with self.assertRaises(VerificationError):
            reps.spinor_characters_su2(group(),(1,0,0,0))

    def test_tensor_invariant_dimensions(self):
        self.assertEqual(reps.tensor_invariants(group()),[1,1,1,1,1,2,3,3,3,4,5,6,7])
        self.assertEqual(reps.tensor_invariants(group(),0),[1])

    def test_bad_partition_dimension(self):
        with self.assertRaises(VerificationError):
            reps.real_partitions(group(),-1)

    def test_principal_and_regular_centralizers(self):
        r = wilson.build()
        self.assertEqual(r['fixed_adjoint_dimensions'],{
            'principal_E8':0,'principal_E6':0,'principal_E6_in_E8':14,
            'regular_A1_in_E6_in_E8':133,'regular_A1_in_E8':133})

    def test_natural_j32_type_certificate(self):
        r = wilson.natural_j32()
        self.assertEqual((r['D8_fixed_dimension'],r['spinor_fixed_dimension'],r['E8_fixed_dimension']),
                         (31,24,55))
        self.assertEqual((r['E8_fixed_lie_type'],r['rank']),('B5 = so(11)',5))

    def test_bad_branching_fails(self):
        with self.assertRaises(VerificationError):
            wilson.fixed({0:8,1:1},248)

    def test_crt_exhaustive_ring_check(self):
        self.assertEqual(gamma.verify_crt()['ring_pairs_checked'],4900)

    def test_invalid_crt(self):
        with self.assertRaises(VerificationError):
            gamma.crt(2,0,0)

    def test_modular_dimensions_small_levels(self):
        self.assertEqual((gamma.principal_modular(3)['genus'],gamma.principal_modular(3)['cusps']),(0,4))
        self.assertEqual(gamma.principal_modular(5)['dimensions']['2']['modular'],11)
        self.assertEqual(gamma.principal_modular(7)['genus'],3)

    def test_modular70_and140(self):
        r = gamma.principal_modular(70)
        self.assertEqual((r['cusps'],r['genus'],r['dimensions']['2']['modular']),(1728,9217,10944))
        self.assertEqual(gamma.principal_modular(140)['dimensions']['2']['modular'],84096)

    def test_modular_exceptions_guarded(self):
        for n in (1,2):
            with self.assertRaises(VerificationError):
                gamma.principal_modular(n)

    def test_gamma0_newspace(self):
        r = gamma.new_weight2(70)
        self.assertEqual((r[14],r[35],r[70]),(1,3,1))

    def test_sl27_exact_character_field(self):
        r = character_data(7)
        self.assertEqual((r['order'],r['exponent']),(336,168))
        self.assertEqual(gamma.character_field(r)['Gamma70_field_degree'],8)

    def test_gamma_full_certificate(self):
        r = gamma.build()
        self.assertEqual(r['exponent'],840)
        self.assertEqual(r['fs_counts'],{'-1':93,'0':108,'1':96})

    def test_delta_exact_irreducibility(self):
        for n in (3,5):
            r = eclectic.delta_certificate(n)
            self.assertEqual((r['order'],r['character_norm']),(6*n*n,1))

    def test_nonsplit_extension_counterexample(self):
        r = eclectic.nonsplit_s3_extension()
        self.assertFalse(r['split'])
        self.assertEqual(r['transposition_lift_orders'],[4,4])

    def test_twisted_c2_refutes_old_cap(self):
        r = eclectic.cyclic_twist_counterexample()
        self.assertEqual((r['base_group_exponent'],r['twist_order']),(2,4))

    def test_cross_prime_irreducible(self):
        self.assertEqual(eclectic.cross_prime_counterexample()['degree'],6)

    def test_product_dimensions(self):
        r = compact.build()
        self.assertEqual([x['real_dimension'] for x in r['topologies'][2:]],[5,5,5,7])
        self.assertIsNone(r['summary']['all_compactifications_excluded'])

    def test_free_quotient_requires_proof_and_divisibility(self):
        for chi,n,proof in [(24,120,True),(-6,120,True),(0,120,False),(0,0,True)]:
            with self.assertRaises(VerificationError):
                compact.free_quotient_euler(chi,n,free_action_verified=proof)

    def test_z2_quotient_generation_arithmetic(self):
        chi = compact.free_quotient_euler(-12,2,free_action_verified=True)
        self.assertEqual(compact.standard_embedding_net_generations(chi),3)


class InterfaceTests(unittest.TestCase):
    def test_nan_is_not_json(self):
        with self.assertRaises(ValueError):
            encode({'bad':float('nan')})

    def test_unsupported_types_not_stringified(self):
        with self.assertRaises(TypeError):
            encode({'bad':{1,2}})

    def test_atomic_write_and_failure_preserve_prior_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'out.json'
            atomic_write(p,'old\n')
            with patch('verification_io.os.replace',side_effect=OSError('injected replace failure')):
                with self.assertRaises(OSError):
                    atomic_write(p,'new\n')
            self.assertEqual(p.read_text(),'old\n')
            self.assertEqual(list(Path(d).iterdir()),[p])

    def test_cli_write_check_mismatch_crlf_and_repeat(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            p = Path(d)/'out.json'
            callback = lambda:{'count':1}
            script = str(ROOT/'scripts'/'supplement_verify.py')
            self.assertEqual(cli(callback,script,['--output',str(p)]),0)
            original = p.read_bytes()
            for _ in range(2):
                self.assertEqual(cli(callback,script,['--check',str(p)]),0)
            p.write_bytes(original.replace(b'\n',b'\r\n'))
            self.assertEqual(cli(callback,script,['--check',str(p)]),1)
            p.write_text('{}\n')
            self.assertEqual(cli(callback,script,['--check',str(p)]),1)
            self.assertEqual(p.read_text(),'{}\n')

    def test_failed_build_does_not_publish_pass(self):
        def bad():
            require(False,'injected failure')
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            p = Path(d)/'out.json'
            self.assertEqual(cli(bad,__file__,['--output',str(p)]),1)
            self.assertFalse(p.exists())

    def test_optimized_interpreter_keeps_checks(self):
        env = dict(os.environ,PYTHONPATH=str(ROOT/'scripts'))
        r = subprocess.run([sys.executable,'-O','-c','from exact_2i import require; require(False,"sentinel")'],
                           env=env,capture_output=True,text=True,timeout=10)
        self.assertNotEqual(r.returncode,0)
        self.assertIn('sentinel',r.stderr)

    def test_imports_have_no_output_or_result_writes(self):
        code = '; '.join('import '+name for name in (
            'supplement_verify','verify_2I_reps','wilson_line_centralizer',
            'gamma70_fusion','eclectic_nonsplit','compactification_search'))
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ,PYTHONPATH=str(ROOT/'scripts'))
            r = subprocess.run([sys.executable,'-c',code],cwd=d,env=env,
                               capture_output=True,text=True,timeout=15)
            self.assertEqual((r.returncode,r.stdout,r.stderr),(0,'',''))
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_external_cwd_and_hashseed_repeatability(self):
        with tempfile.TemporaryDirectory() as d:
            contents = []
            for seed in ('1','98765'):
                out = Path(d)/('result-'+seed+'.json')
                r = subprocess.run([sys.executable,str(ROOT/'scripts'/'supplement_verify.py'),
                                    '--output',str(out)],cwd=d,
                                   env=dict(os.environ,PYTHONHASHSEED=seed),
                                   capture_output=True,text=True,timeout=30)
                self.assertEqual(r.returncode,0,r.stderr)
                contents.append(out.read_bytes())
            self.assertEqual(*contents)


if __name__ == '__main__':
    unittest.main()
