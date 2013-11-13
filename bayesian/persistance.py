'''Handle Persistance of Pre-generated Samples'''
import os
import uuid
import sqlite3

from collections import defaultdict

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


def pack_key(key):
    '''In this version we will just
    pack the Booleans to a string consisting
    of 0s and 1s'''
    return ''.join(['1' if k[1] else '0' for k in key])


def unpack_key(key):
    return tuple([('_', x == '1') for x in key])


def pack_data(x):
    return '%2.8f' % x


def unpack_data(x):
    # TODO Change this to use struct module
    return float(x)


def key_to_int(key):
    '''Convert a tuple of Booleans
    into an integer assuming the
    booleans represent bits.
    e.g. (True, False, True) -> 5

    Can we expand this to any discrete
    variable keys? mmm
    What if we had something like
    (boolean, non-boolean, boolean)?

    '''


    retval  = 0
    for i, (k, v) in enumerate(list(key)[::-1]):
        print i, k, v
        if v:
            retval += 2 ** i
    return retval


BOOLEANS = frozenset([False, True])


class DiskArray(object):
    '''Non thread-safe Persistant Array'''

    def __init__(self, default_constructor=None):
        self.name = 'tmp/%s.diskarray' % uuid.uuid4().hex
        self.default_constructor = default_constructor
        self._db = open(self.name, 'w+b')
        if default_constructor is not None:
            self.d = defaultdict(default_constructor)
        else:
            self.d = dict()
        self.record_size = -1 # Until we get the first setitem we
                              # wont know what the actual size is
        self.last_row_num = -1  # So that we can lazily fill in dummy rows...


    def __getitem__(self, key):
        '''For now we only allow dicts
        where the keys are all boolean
        to persist.'''
        #row_num = key_to_int(key)
        row_num = key
        if row_num > self.last_row_num:
            raise KeyError(key)
        offset = row_num * self.record_size
        self._db.seek(offset)
        row = self._db.read(self.record_size)
        #w_key, val = row.split(':')
        val = row.strip()
        if row.startswith('D'):
            if self.default_constructor:
                value = self.default_constructor()
                self[key] = value
                return value
            else:
                raise KeyError(key)
        return float(val)

    def build_record(self, key, value):
        '''Build a single row for the
        underlying fixed length file'''
        row = '%s\n' % (pack_data(value))
        if self.record_size == -1:
            self.record_size = len(row)
            self.dummy_record = 'D' * (self.record_size - 1)  + '\n'
        return row

    def __setitem__(self, key, value):
        row = self.build_record(key, value)
        row_num = key
        if row_num > self.last_row_num:
            # We need to write dummy
            # rows to reserve space
            for offset in range(
                    (self.last_row_num + 1) * self.record_size,
                    row_num * self.record_size,
                    self.record_size):
                self._db.seek(offset)
                self._db.write(self.dummy_record)
        if self.record_size != -1:
            offset = row_num * self.record_size
            self._db.seek(offset)
        self._db.write(row)
        self._db.flush()
        if row_num > self.last_row_num:
            self.last_row_num = row_num
        #return value

    def __iter__(self):
        self._db.seek(0)
        for row_num in range(0, self.last_row_num + 1):
            row = self._db.read(self.record_size)
            value = row.strip()
            if value.startswith('D'):
                yield None
            else:
                yield float(value)


    def __len__(self):
        return self.last_row_num


    #def values(self):
    #    return self.d.values()
    #    #for k in self._db.keys():
    #    #    yield unpack(self._db[k])

    #def items(self):
    #    if self.d:
    #        return self.d.items()
    #    return list(self.iteritems())



    def iteritems(self):
        if self.d:
            for k, v in self.d.iteritems():
                yield k, v
        else:
            for row_num, value in zip(
                    range(0, self.last_row_num + 1),
                    super(PersistantTT, self).__iter__()):
                offset = row_num * self.record_size
                row = self._db.read(self.record_size)
                key, value = row.split(':')
                if key.startswith('D'):
                    continue
                yield unpack_key(key), float(value)



    def __contains__(self, key):
        try:
            val = self[key]
        except KeyError:
            return False
        return True


    def copy(self):
        c = DiskArray(self.default_constructor)
        self._db.seek(0)
        for row_num in range(0, self.last_row_num + 1):
            offset = row_num * self.record_size
            row = self._db.read(self.record_size)
            #key, value = row.split(':')

            if row.startswith('D'):
                continue
            c[row_num] = unpack_data(row.strip())
        return c

    #def __del__(self):
    #    self._db.close()
    #    os.unlink(self.name)

class PersistantTT(DiskArray):
    '''
    A class which acts as like a dictionary
    but uses a DiskDict as a store instead
    of in memory. Since the DiskDict uses
    a fixed width record format we have
    to ensure that all the keys and values
    are the same length. This class
    will perform key transposes when
    setting and getting items from
    the DiskDict by transposing a
    key of (('a', True'), ('b', 'False'))
    into '10' for efficient storage.
    '''
    def __init__(self, defaultconstructor=None):
        super(PersistantTT, self).__init__(defaultconstructor)
        self.all_boolean = None
        self.variable_names = []

    def _pack_key(self, key):
        '''In this version we will just
        pack the Booleans to a int'''
        # First record the variable names so
        # that we can reconstruct the keys later
        # for example during iteritems, keys etc
        if not self.variable_names:
            self.variable_names = [k[0] for k in key]
        assert [k[0] for k in key] == self.variable_names
        return key_to_int(key)


    def _unpack_key(self, key):
        return [(k[0], k[1]=='1') for k in zip(
            self.variable_names, list(bin(key)[2:]))]


    def __setitem__(self, key, value):
        if set([k[1] for k in key]).difference(BOOLEANS):
            # Means there are non-booleans in the key
            # TODO: Allow non-booleans
            raise "Not Implemented"
        super(PersistantTT, self).__setitem__(
            self._pack_key(key), value)

    def __getitem__(self, key):
        if self._pack_key(key) > self.last_row_num:
            if self.default_constructor:
                value = self.default_constructor()
                super(PersistantTT, self).__setitem__(
                    self._pack_key(key), value)
                return value
            else:
                raise KeyError(key)
        return super(PersistantTT, self).__getitem__(
            self._pack_key(key))

    #def iteritems(self):
    #    if self.d:
    #        for k, v in self.d.iteritems():
    #            yield k, v
    #    else:
    #        for i, v in super(PersistantTT, self):
    #        self._db.seek(0)
    #        for row_num in range(0, self.last_row_num + 1):
    #            offset = row_num * self.record_size
    #            row = self._db.read(self.record_size)
    #            value = row.strip()
    #            if value.startswith('D'):
    #                continue
    #            yield self._unpack_key(row_num), float(value)
