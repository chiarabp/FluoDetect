from DB import DB
import unittest


class TestMyModule(unittest.TestCase):
    def test(self):
        db = DB('prova')
        db.look_up_records_pks_wav_fwhm(514,20,40,100)

if __name__ == '__main__':
    unittest.main()