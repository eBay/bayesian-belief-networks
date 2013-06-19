'''Handle Persistance of Pre-generated Samples'''
import sqlite3

class UnsupportedTypeException(Exception):
    pass


def build_row_factory(conn):
    '''
    Introspect the samples table
    to build the row_factory
    function. We will assume that
    numeric values are Boolean
    and all other values are Strings.
    Should we encounter a numeric
    value not in (0, 1) we will
    raise an error.
    '''
    cur = conn.cursor()
    cur.execute("pragma table_info('data')")
    cols = cur.fetchall()
    column_metadata = dict([(col[1], col[2]) for col in cols])

    def row_factor(cursor, row):
        row_dict = dict()
        for idx, desc in enumerate(cursor.description):
            col_name = desc[0]
            col_val = row[idx]
            if column_metadata[col_name] == 'integer':
                row_dict[col_name] = col_val == 1
            elif column_metadata[col_name] == 'varchar':
                row_dict[col_name] = col_val
            elif column_metadata[col_name] == 'text':
                row_dict[col_name] = col_val
            else:
                raise UnsupportedTypeException
        return row_dict

    return row_factor



class SampleDB(object):

    def __init__(self, filename):
        self.conn = sqlite3.connect(filename)
        self.conn.row_factory = build_row_factory(self.conn)
        self.insert_count = 0

    def get_samples(self, n, **kwds):
        self.commit()
        cur = self.conn.cursor()
        #sql = '''
        #    SELECT * FROM samples
        #'''
        sql = '''
            SELECT * FROM data
        '''
        evidence_cols = []
        evidence_vals = []
        for k, v in kwds.items():
            evidence_cols.append('%s=?' % k)
            if isinstance(v, bool):
                # Cast booleans to integers
                evidence_vals.append(int(v))
            else:
                evidence_vals.append(v)
        if evidence_vals:
            sql += '''
                WHERE %s
            ''' % ' AND '.join(evidence_cols)
        sql += ' LIMIT %s' % n
        cur.execute(sql, evidence_vals)
        return cur.fetchall()

    def save_sample(self, sample):
        '''
        Given a dict representing
        a sample save it to the sqlite db
        with default type mapping.
        The sqlite3 module automatically
        converts booleans to integers.
        '''
        import ipdb; ipdb.set_trace()
        keys, vals = zip(*sample.items())
        sql = '''
            INSERT INTO SAMPLES
            (%(columns)s)
            VALUES
            (%(values)s)
        ''' % dict(
            columns=', '.join(keys),
            values=', '.join(['?'] * len(vals)))
        cur = self.conn.cursor()
        cur.execute(sql, vals)
        self.insert_count += 1
        self.commit()

    def commit(self):
        if self.insert_count >= 1000:
            print 'Committing....'
            try:
                self.conn.commit()
                self.insert_count == 0
            except:
                print 'Commit to db file failed...'
                raise
