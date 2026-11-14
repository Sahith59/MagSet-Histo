"""Bounded synthetic controls; these are not research performance measurements."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from magset import multicrop_compare as m
from magset.bags import fit_normalization
from magset.data import CLASS_NAMES


class MulticropCompareTests(unittest.TestCase):
    def config(self):
        return {'thumbnail_weight': .5, 'quadrant_weight': .5, 'fixed_C': .01, 'max_iter': 1000,
                'normalization': 'field-moments-v2', 'encoder': {'dimension': 1024},
                'manifest_sha256': 'manifest', 'protocol_sha256': 'protocol'}

    def toy(self):
        counts = [4,38,10,5,9,6,3,7]; outer = [2,12,3,2,3,2,1,3]
        cases = []; roles = {}; bags = []
        for label, (count, held) in enumerate(zip(counts, outer)):
            for j in range(count):
                cid = '%d-%02d'%(label,j)
                cases.append({'case_id': cid, 'label': label})
                roles[cid] = 'test' if j < held else 'train'
                fields = np.asarray([[label, j/10], [label+.1, j/10+.1]], dtype=np.float32)
                bags.append([fields.copy() for _ in range(4)])
        trusted = {'unit': 'provider-case', 'manifest_sha256': 'manifest', 'protocol_sha256': 'protocol',
                   'cases': cases, 'labels': {c['case_id']: c['label'] for c in cases},
                   'trials': {str(t): {'case_roles': roles.copy()} for t in range(1,6)}}
        return cases, bags, trusted

    def test_fixed_settings(self):
        m.validate_config(self.config())
        for key, value in [('fixed_C',1), ('max_iter',2000), ('normalization','case-mean-v1')]:
            changed = self.config(); changed[key] = value
            with self.assertRaises(ValueError): m.validate_config(changed)

    def test_field_moments_not_case_mean_std(self):
        bags = [[np.asarray([[0.],[10.]],dtype=np.float32)]*4,
                [np.asarray([[4.]],dtype=np.float32)]*4]
        mean, std = fit_normalization(bags,[0,1])
        np.testing.assert_allclose(mean,4.5)
        np.testing.assert_allclose(std,np.sqrt(12.75))
        self.assertNotEqual(float(std[0,0]), .5)

    def test_one_run_replay_roles_and_no_outer(self):
        cases,bags,trusted = self.toy()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'run'
            report,_ = m.run_one(folder,bags,cases,trusted,1,812,self.config(),{}, {})
            self.assertTrue(report['converged']); self.assertEqual(report['fixed_C'], .01)
            self.assertEqual([len(report['case_ids'][k]) for k in ('train','validation','test')],[46,8,28])
            records=[json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),120)
            self.assertFalse(set(r['case_id'] for r in records)&set(report['case_ids']['test']))
            z=np.asarray([[x.mean(0) for x in c] for c in bags])
            with self.assertRaises(ValueError):m.score_saved(folder/'model.npz',z,[0],cases,'test')
            # An independently changed saved coefficient changes replay probabilities.
            with np.load(folder/'model.npz',allow_pickle=False) as a: saved={k:a[k] for k in a.files}
            saved['coef_0']=saved['coef_0'].copy();saved['coef_0'][0,0]+=1
            np.savez(folder/'changed.npz',**saved)
            indices=[i for i,c in enumerate(cases) if c['case_id'] in report['case_ids']['validation']]
            changed=m.score_saved(folder/'changed.npz',z,indices,cases,'validation')
            self.assertFalse(np.allclose([r['probabilities'] for r in changed],[r['probabilities'] for r in records]))

    def test_exact_twenty_matrix_same_C(self):
        cases,bags,trusted=self.toy(); config=self.config()
        ref={'normalization_policy':'field-moments-v2','method':'mean','encoder':'phikon-v2',
             'feature_provenance':[{'feature_sha256':str(i)} for i in range(4)], 'selected_C':1,
             'validation':{}}
        real_verify=m.verify_run
        def verify(path, context):
            if 'reference' in str(path):
                value=dict(ref); name=Path(path).parent.name
                value['seed']=int(name.split('seed')[1]);value['trial']=int(name.split('trial')[1].split('-')[0])
                return value, {'metrics_sha256':'historical'}
            return real_verify(path,context)
        with tempfile.TemporaryDirectory() as tmp:
            cfg=Path(tmp)/'config.json';cfg.write_text(json.dumps(config))
            with patch.object(m,'load_trusted_context',return_value=trusted), patch.object(m,'read_manifest',return_value=[]),\
                 patch.object(m,'load_inputs',return_value=(cases,{'thumbnail':bags,'multicrop':bags},ref['feature_provenance'],[])),\
                 patch.object(m,'file_hash',side_effect=lambda p:'manifest' if str(p)=='manifest' else 'hash'),\
                 patch.object(m,'verify_run',side_effect=verify):
                result=m.compare('manifest','protocol','thumbnail','multicrop','reference',cfg,Path(tmp)/'out','a'*40)
            self.assertEqual(result['fits'],20);self.assertEqual(len(result['runs']),20)
            self.assertFalse(result['gate_passed'])
            for seed in ('812','813'):self.assertEqual(result['summary'][seed]['delta_f1'],0)

    def test_cache_formula_identity_and_bounds(self):
        from magset.data import file_hash
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); old=root/'old';new=root/'new';old.mkdir();new.mkdir()
            config=self.config();config['thumbnail_provenance']={};rows=[];om=[];nm=[]
            for scale in m.MAGNIFICATIONS:
                p=str(scale)+'.png';g=np.ones((1,1024),np.float32);q=np.ones((1,4,1024),np.float32)*2
                fields={'paths':np.asarray([p]),'features':np.asarray(.5*g.astype(np.float64)+.5*q.mean(1,dtype=np.float64),dtype=np.float32),
                        'thumbnail_features':g,'quadrant_features':q,'raw_sha256':np.asarray(['raw']),
                        'widths':np.asarray([4]),'heights':np.asarray([4]),
                        'bounds':np.asarray([[[0,0,2,2],[2,0,4,2],[0,2,2,4],[2,2,4,4]]])}
                fields={key:np.concatenate([value,value],axis=0) for key,value in fields.items()}
                fields['paths']=np.asarray([p,p+'.second'])
                rows.extend({'relative_path':path,'width':'4','height':'4','sha256':'raw'} for path in fields['paths'].tolist())
                op=old/('phikon-v2-%d.npz'%scale);np.savez(op,paths=fields['paths'],features=np.concatenate([g,g]))
                meta={'encoder':config['encoder'],'feature_sha256':file_hash(op),'transform':{'resize':224}}
                op.with_suffix('.json').write_text(json.dumps(meta));om.append(meta)
                pins={'npz_sha256':file_hash(op),'json_sha256':file_hash(op.with_suffix('.json'))}
                config['thumbnail_provenance'][str(scale)]=pins
                cp=new/op.name;np.savez(cp,**fields)
                nm.append({'encoder':config['encoder'],'thumbnail_metadata':meta,'thumbnail_provenance':pins,
                           'thumbnail_feature_sha256':meta['feature_sha256'],'config_sha256':'cfg',
                           'code_sha':'a'*40,'archive_sha256':'b'*64,'source_sha256':{'source':'c'*64},
                           'transform':meta['transform'], 'config':config, 'images':2, 'smoke':False,
                           'smoke_limit':0, 'smoke_roles':None, 'quadrant_forwards':8,
                           'technical_thumbnail_forwards':0, 'blend_replay_max_abs':0,
                           'thumbnail_parity_max_abs':None,
                           'blend':'float32(0.5*float64(g) + 0.5*mean(q1..q4,float64))'})
            cases=[{'case_id':'toy'}]
            def loads(*args):return (cases,[],om if Path(args[1])==old else nm)
            real_load=np.load
            class ReadOnceArchive:
                def __init__(self,*args,**kwargs):self.archive=real_load(*args,**kwargs);self.reads=set()
                @property
                def files(self):return self.archive.files
                def __enter__(self):return self
                def __exit__(self,*args):self.archive.close()
                def __getitem__(self,key):
                    if key in self.reads:raise AssertionError('Repeated compressed-array decompression: '+key)
                    self.reads.add(key);return self.archive[key]
            with patch.object(m,'load_bags',side_effect=loads), patch.object(m.np,'load',side_effect=ReadOnceArchive):
                contract={k:nm[0][k] for k in ('code_sha','archive_sha256','source_sha256')}
                m.load_inputs(rows,old,new,'manifest',config,'cfg',contract)
                for key,value in [('code_sha','f'*40), ('archive_sha256','f'*64), ('source_sha256',{'other':'f'*64}), ('quadrant_forwards',5), ('smoke',True), ('technical_thumbnail_forwards',1)]:
                    previous=[meta[key] for meta in nm]
                    for meta in nm:meta[key]=value
                    with self.assertRaises(ValueError):m.load_inputs(rows,old,new,'manifest',config,'cfg',contract)
                    for meta,old_value in zip(nm,previous):meta[key]=old_value
                cp=new/'phikon-v2-40.npz'
                with np.load(cp,allow_pickle=False) as a:values={k:a[k] for k in a.files}
                values['thumbnail_features']=values['thumbnail_features']+1
                np.savez(cp,**values)
                with self.assertRaisesRegex(ValueError,'thumbnail differs'):
                    m.load_inputs(rows,old,new,'manifest',config,'cfg')

    def test_stable_underflow_loss_and_complete_metrics(self):
        cases=[{'case_id':str(i),'label':i} for i in range(8)]
        z=np.zeros((8,4,2),dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.npz'
            np.savez(path,mean=np.zeros((4,2),np.float32),std=np.ones((4,2),np.float32),
                     coef_0=np.zeros((8,6)),intercept_0=np.asarray([0.,-1000.,0.,0.,0.,0.,0.,0.]),classes_0=np.arange(8))
            records=m.score_saved(path,z,np.arange(8),cases,'validation')
            value=m.summarize(records)
            self.assertEqual(records[1]['probabilities'][1],0)
            self.assertGreater(-records[1]['log_probabilities'][1],1000)
            self.assertGreater(value['mean_subset_cross_entropy'],125)
            self.assertEqual(value['distinct_class_support'],[1]*8)
            self.assertEqual(value['distinct_cases'],8)
            self.assertEqual(len(value['all_view']['confusion_matrix']),8)
            self.assertEqual(len(value['all_view']['class_recall']),8)
            self.assertEqual(value['pooled']['records'],120)
            self.assertAlmostEqual(value['pooled_class_balanced']['weight_sum'],1)
            self.assertAlmostEqual(value['mean_subset_cross_entropy'],value['mean_subset_class_balanced_cross_entropy'])

    def test_original_feature_snapshot_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp);(source/'configs').mkdir()
            config=source/'configs/breakhis-multicrop-v1.json';config.write_text('{}')
            (source/'.source-sha').write_text('a'*40)
            (source/'.archive-sha256').write_text('b'*64)
            for name in m.EXTRACTOR_SOURCE_FILES:
                path=source/name;path.parent.mkdir(exist_ok=True);path.write_text(name)
            contract=m.feature_source_contract(source,config)
            self.assertEqual(contract['code_sha'],'a'*40)
            self.assertEqual(set(contract['source_sha256']),set(m.EXTRACTOR_SOURCE_FILES))
            other=source/'different.json';other.write_text('{"changed":true}')
            with self.assertRaises(ValueError):m.feature_source_contract(source,other)
            (source/'.source-sha').write_text('invalid')
            with self.assertRaises(ValueError):m.feature_source_contract(source,config)

    def test_invalid_partition_prevents_fit(self):
        cases,bags,trusted=self.toy()
        trusted['trials']['1']['case_roles'][cases[0]['case_id']]='train'
        with tempfile.TemporaryDirectory() as tmp, patch.object(m,'fit_models') as fit:
            with self.assertRaises(ValueError):m.run_one(Path(tmp)/'run',bags,cases,trusted,1,812,self.config(),{}, {})
            fit.assert_not_called()


if __name__=='__main__':unittest.main()
