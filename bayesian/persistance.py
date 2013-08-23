'''Handle Persistance of Pre-generated Samples'''
import os
import sqlite3


class UnsupportedTypeException(Exception):
    pass


class SampleDBNotFoundException(Exception):
    pass


COMMIT_THRESHOLD = 1000

# Python data type to SQLite data type mapping
# NOTE: Technically SQLite does not support
# boolean types, they are internally stored
# as 0 and 1, however you can still issue
# a create statement with a type of 'bool'.
# We will use this to distinguish between
# boolean and integer data types.
P2S_MAPPING = {
    bool: 'bool',
    str: 'varchar',
    unicode: 'varchar',
    int: 'integer'}


S2P_MAPPING = {
    'bool': bool,
    'varchar': unicode,
    'integer': int}


def domains_to_metadata(domains):
    '''Construct a metadata dict
    out of the domains dict.
    The domains dict has the following
    form:
    keys: variable names from a factor graph
    vals: list of possible values the variable can have
    The metadata dict has the following form:
    keys: (same as above)
    vals: A string representing the sqlite data type
    (i.e 'integer' for bool and 'varchar' for str)'''
    metadata = dict()
    for k, v in domains.items():
        # Assume that all values in the domain
        # are of the same type. TODO: verify this!
        try:
            metadata[k.name] = P2S_MAPPING[type(v[0])]
        except KeyError:
            print k, v
            raise UnsupportedTypeException
    return metadata


def ensure_data_dir_exists(filename):
    data_dir = os.path.dirname(filename)
    if not os.path.exists(data_dir):
        # Create the data directory...
        os.makedirs(data_dir)


def initialize_sample_db(conn, metadata):
    '''Create a new SQLite sample database
    with the appropriate column names.
    metadata should be a dict of column
    names with a type. Currently if
    the Variable is a boolean variable
    we map it to integers 1 and 0.
    All other variables are considered
    to be categorical and are mapped
    to varchar'''
    type_specs = []
    for column, sqlite_type in metadata.items():
        type_specs.append((column, sqlite_type))
    SQL = '''
        CREATE TABLE samples (%s);
    ''' % ','.join(['%s %s' % (col, type_) for col, type_ in type_specs])
    cur = conn.cursor()
    print SQL
    cur.execute(SQL)


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
    cur.execute("pragma table_info('samples')")
    cols = cur.fetchall()
    column_metadata = dict([(col[1], col[2]) for col in cols])

    def row_factory(cursor, row):
        row_dict = dict()
        for idx, desc in enumerate(cursor.description):
            col_name = desc[0]
            col_val = row[idx]
            try:
                row_dict[col_name] = \
                    S2P_MAPPING[column_metadata[col_name]](col_val)
            except KeyError:
                raise UnsupportedTypeException(
                    'A column in the SQLite samples '
                    'database has an unsupported type. '
                    'Supported types are %s. ' % str(S2P_MAPPING.keys()))
        return row_dict

    return row_factory


class SampleDB(object):

    def __init__(self, filename, domains, initialize=False):
        self.conn = sqlite3.connect(filename)
        self.metadata = domains_to_metadata(domains)
        if initialize:
            initialize_sample_db(self.conn, self.metadata)
        self.conn.row_factory = build_row_factory(self.conn)
        self.insert_count = 0

    def get_samples(self, n, **kwds):
        cur = self.conn.cursor()
        sql = '''
            SELECT * FROM samples
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
        Given a list of tuples
        (col, val) representing
        a sample save it to the sqlite db
        with default type mapping.
        The sqlite3 module automatically
        converts booleans to integers.
        '''
        #keys, vals = zip(*sample.items())
        keys = [x[0] for x in sample]
        vals = [x[1] for x in sample]
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
        if self.insert_count >= COMMIT_THRESHOLD:
            self.commit()

    def commit(self):
        print 'Committing....'
        try:
            self.conn.commit()
            self.insert_count = 1
        except:
            print 'Commit to db file failed...'
            raise
