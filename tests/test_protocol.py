import unittest
from magset.data import case_id, cases_from_rows
from magset.protocol import audit_fold


def row(subtype, code, raw, mag):
    return {'filename': 'SOB_%s_%s-%s-%s-001.png' % ('B' if code in {'A','F','PT','TA'} else 'M',code,raw,mag),
            'case_id_raw':raw,'patient_group_conservative':raw.rstrip('ABCD'),
            'subtype':subtype,'magnification':str(mag),'relative_path':'%s/%s/%s.png'%(code,raw,mag)}


class ProtocolTests(unittest.TestCase):
    def test_same_raw_id_under_different_labels_is_not_one_case(self):
        rows=[row('ductal_carcinoma','DC','14-1234',40),row('lobular_carcinoma','LC','14-1234',40)]
        self.assertNotEqual(case_id(rows[0]),case_id(rows[1]))
        self.assertEqual(len(cases_from_rows(rows)),2)

    def test_stem_overlap_is_reported_without_changing_provider_ids(self):
        rows=[row('adenosis','A','14-1234AB',40),row('adenosis','A','14-1234CD',100)]
        text='\n'.join(r['filename']+'|'+r['magnification']+'|1|'+role for r,role in zip(rows,['train','test']))
        report=audit_fold(text,rows)
        self.assertEqual(report['numeric_stem_overlap'],1)
        self.assertEqual(report['provider_case_overlap'],0)

    def test_case_cannot_cross_roles_or_omit_images(self):
        rows=[row('adenosis','A','14-1234',40),row('adenosis','A','14-1234',100)]
        text='\n'.join(r['filename']+'|'+r['magnification']+'|1|'+role for r,role in zip(rows,['train','test']))
        with self.assertRaises(ValueError):audit_fold(text,rows)
        with self.assertRaises(ValueError):audit_fold(text.splitlines()[0],rows)


if __name__=='__main__':unittest.main()
