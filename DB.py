import os

import pandas as pd
import numpy as np
from FluoRec import FluoRec
import sqlite3
from pathlib import Path
import bcrypt


class DB(object):
    def __init__(self, db_name):
        self.con = sqlite3.connect("fluodb.sqlite")
        self.cur = self.con.cursor()
        self.db_name = db_name
        # self.db = pd.read_csv('general_db.txt', delimiter=";", header=0)
        # self.db.columns = ['id', 'Name', 'pks_num', 'pks_wav', 'fwhm']
        pass

    @staticmethod
    def login(user, passw):
        salt = bcrypt.gensalt()

        con = sqlite3.connect("fluodb.sqlite")
        df = pd.read_sql_query(f"SELECT id FROM Profiles WHERE username='{user} AND password='{bcrypt.hashpw(passw.encode('utf-8'), salt)}'", con)

        if df.empty:
            return 0
        return df['id']

    @staticmethod
    def get_databases_list():
        con = sqlite3.connect("fluodb.sqlite")
        cur = con.cursor()
        df = pd.read_sql_query("SELECT name from sqlite_master WHERE type='table'", con)
        df = df[df['name'] != 'Profiles']

        # number of records per db
        table_rec_nums = np.zeros(len(df))

        for index, row in df.iterrows():
            # number of records per db
            df_rec_nums = pd.read_sql_query(f"SELECT DISTINCT id FROM {row['name']}", con)
            table_rec_nums[index] = len(df_rec_nums)

        con.close()

        return df, table_rec_nums

    @staticmethod
    def create_db(table_name):
        con = sqlite3.connect("fluodb.sqlite")
        cur = con.cursor()

        df = pd.read_sql_query("SELECT name from sqlite_master WHERE type='table'", con)

        if table_name in df['name'].values:
            return 0 # table with name already exists!

        query = f"CREATE TABLE {table_name} (id INTEGER, Name TEXT, pks_num INTEGER, pks_wav REAL, fwhm REAL, min_peak_height REAL, dangerous INTEGER);"
        cur.execute(query)
        con.commit()
        con.close()

        return 1

    def get_db(self):
        df = pd.read_sql_query(f"SELECT DISTINCT id FROM {self.db_name}", self.con)
        df.columns = ['id']

        columns_union = ['id', 'Name', 'pks_num', 'pks_wav', 'fwhm', 'min_peak_height']
        df_union = pd.DataFrame(columns=columns_union)

        for row_id in df['id'].values:
            df_id = pd.read_sql_query(f"SELECT id, Name, pks_num, pks_wav, fwhm, min_peak_height FROM {self.db_name} where id='{row_id}'", self.con)
            df_id.columns = columns_union

            pks_wav = np.zeros(len(df_id))
            fwhm = np.zeros(len(df_id))

            for index, row in df_id.iterrows():
                pks_wav[index] = row['pks_wav']
                fwhm[index] = row['fwhm']

            # union of all the records with same id (different peak values)
            df_union.loc[len(df_union)] = {'DB Name': self.db_name, 'id': row_id, 'Name': df_id['Name'][0], 'pks_num': df_id['pks_num'][0],
             'pks_wav': pks_wav, 'fwhm': fwhm, 'min_peak_height': df_id['min_peak_height'][0]}

        return df_union

    def add_record(self, name, pks_wavs, fwhm, min_peak_height):
        df = pd.read_sql_query(f"SELECT id FROM {self.db_name}", self.con)
        df.columns = ['id']

        ids = df['id'].array

        id_ = 1
        if ids:
            id_ = int(ids[-1]) + 1 # last element
        else:
            id_ = 1

        for index, pks_wav in enumerate(pks_wavs):
            query = f"INSERT INTO {self.db_name} (id, Name, pks_num, pks_wav, fwhm, min_peak_height, dangerous) VALUES ({id_}, '{name}', {len(pks_wavs)}, {float("{:.2f}".format(pks_wav))}, {float("{:.2f}".format(fwhm[index]))}, {min_peak_height}, {0})"
            self.cur.execute(query)
            self.con.commit()

        return id_

    def look_up_by_id(self, id_):
        df = pd.read_sql_query(f"SELECT id, Name, pks_num, pks_wav, fwhm, min_peak_height FROM {self.db_name} where id='{id_}'", self.con)
        if df.empty:
            return df, FluoRec("", -1, "", 0, [], [], 0)  # returns empty dataframe
        df.columns = ['id', 'Name', 'pks_num', 'pks_wav', 'fwhm', 'min_peak_height']

        pks_wav = np.zeros(len(df))
        fwhm = np.zeros(len(df))

        for index, row in df.iterrows():
            pks_wav[index] = row['pks_wav']
            fwhm[index] = row['fwhm']

        # union of all the records with same id (different peak values)
        df_union = pd.DataFrame({'DB Name': self.db_name, 'id': id_, 'Name': df['Name'][0], 'pks_num': df['pks_num'][0],
                                 'pks_wav': [pks_wav], 'fwhm': [fwhm], 'min_peak_height': df['min_peak_height'][0]})

        record = FluoRec(
            self.db_name,
            id_,
            df['Name'][0],
            df['pks_num'][0],
            pks_wav,
            fwhm,
            df['min_peak_height'][0]
        )

        return df_union, record

    def look_up_by_name(self, name):
        # list of all the ids (no duplicates)
        df = pd.read_sql_query(f"SELECT DISTINCT id FROM {self.db_name}", self.con)
        df.columns = ['id']

        columns_union = ['id', 'Name', 'pks_num', 'pks_wav', 'fwhm', 'min_peak_height']
        df_union = pd.DataFrame(columns=columns_union)

        for row_id in df['id'].values:
            df_name = pd.read_sql_query(f"SELECT id, Name, pks_num, pks_wav, fwhm, min_peak_height FROM {self.db_name} where id='{row_id}' AND Name='{name}'", self.con)

            if not df_name.empty:
                df_name.columns = columns_union

                pks_wav = np.zeros(len(df))
                fwhm = np.zeros(len(df))

                for index, row in df_name.iterrows():
                    pks_wav[index] = row['pks_wav']
                    fwhm[index] = row['fwhm']

                # union of all the records with same id (different peak values)
                df_union.loc[len(df_union)] = {'id': row_id, 'Name': df_name['Name'][0], 'pks_num': df_name['pks_num'][0],
                     'pks_wav': pks_wav, 'fwhm': fwhm, 'min_peak_height': df_name['min_peak_height'][0]}

        return df_union

    def look_up_records_pks_wav_fwhm(self, pks_wav, fwhm, tolerance_pks, tolerance_fwhm):
        df = pd.read_sql_query(f"SELECT id, Name, pks_num, pks_wav, fwhm, min_peak_height FROM {self.db_name} "
                               f"WHERE ((pks_wav > {pks_wav - tolerance_pks}) AND (pks_wav < {pks_wav + tolerance_pks})) AND"
                               f" ((fwhm > {fwhm - tolerance_fwhm}) AND (fwhm < {fwhm + tolerance_fwhm}))", self.con)
        columns = ['id', 'Name', 'pks_num', 'pks_wav', 'fwhm', 'min_peak_height']
        df.columns = columns

        df_union = pd.DataFrame(columns=columns)
        df_union['pks_wav'] = df_union['pks_wav'].astype(object)
        df_union['fwhm'] = df_union['fwhm'].astype(object)

        # check first if there is any double record (same id but different wavelength|fwhm)
        # order in ascending order to facilitate previous id search
        df_sorted = df.sort_values(by='id', ascending=True)
        previous_id = None
        for index, row in df_sorted.iterrows():
            if (previous_id is None) or (row['id'] != previous_id):
                previous_id = row['id']
                df_union.loc[len(df_union)] = {'id': previous_id, 'Name': row['Name'],
                                               'pks_num': row['pks_num'],
                                               'pks_wav': [row['pks_wav']], 'fwhm': [row['fwhm']],
                                               'min_peak_height': row['min_peak_height']}
            elif row['id'] == previous_id:
                condition = df_union['id'] == previous_id
                df_union[condition, 'pks_wav'].append(row['pks_wav'])
                df_union[condition, 'fwhm'].append(row['fwhm'])

        # now check if there are any double record in the database which were not in the tolerances described
        # this is to complete the records shown
        for index_union, row_union in df_union.iterrows():
            df2 = pd.read_sql_query(f"SELECT id, Name, pks_num, pks_wav, fwhm, min_peak_height FROM {self.db_name} WHERE id='{row['id']}'", self.con)

            for index, row in df2.iterrows():
                if row['pks_wav'] not in df_union['pks_wav'][index_union]:
                    df_union['pks_wav'][index_union].append(row['pks_wav'])
                    df_union['fwhm'][index_union].append(row['fwhm'])

        return df_union

    def delete_record(self, id_):
        df = pd.read_sql_query(f"SELECT id FROM {self.db_name} WHERE id='{id_}'", self.con)

        # Check if the id exists in the database
        if not df.empty:
            # Delete files
            try:
                os.remove(f'high_res_spectra_db/{self.db_name}/{id_}.txt')
                os.remove(f'descriptions/{self.db_name}/{id_}.txt')
            except FileNotFoundError:
                print('Description or spectra file does not exist')

            query = f"DELETE FROM {self.db_name} WHERE id='{id_}'"
            self.cur.execute(query)
            self.con.commit()

            return 1
        return 0 # record doesnt exist

    @staticmethod
    def get_record_description(db_name, id_):
        file_path = f'descriptions/{db_name}/{id_}.txt'

        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return file.read()
        else:
            return "File not found"

    @staticmethod
    def update_record_description(db_name, id_, description):
        file_path = Path(f'descriptions/{db_name}/{id_}.txt')
        os.makedirs(f'descriptions/{db_name}/', exist_ok=True)

        if not file_path.exists():
            file_path.write_text(description)
        else:
            with file_path.open('a') as f:
                f.write(description)

    def close(self):
        self.con.close()






