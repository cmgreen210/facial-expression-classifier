import numpy as np
from boto.s3.connection import S3Connection
import os
from filechunkio import FileChunkIO
import math
import pandas as pd


_conn = None


def get_connection():
    """

    :return:
    """
    global _conn
    if _conn is None:
        _conn = S3Connection(os.environ['AWS_ACCESS_KEY_ID'],
                             os.environ['AWS_SECRET_ACCESS_KEY'])
    return _conn


def upload_big_file(source_path, bucket):
    """

    :param source_path:
    :param bucket:
    :return:
    """
    source_size = os.stat(source_path).st_size
    mp = bucket.initiate_multipart_upload(os.path.basename(source_path))

    chunk_size = 52428800
    chunk_count = int(math.ceil(source_size / chunk_size))
    for i in range(chunk_count + 1):
        offset = chunk_size * i
        bytes = min(chunk_size, source_size - offset)
        with FileChunkIO(source_path, 'r', offset=offset,
                         bytes=bytes) as fp:
            mp.upload_part_from_file(fp, part_num=i + 1)

    mp.complete_upload()


def _load_original_data_into_df(file_path):
    df = pd.read_csv(file_path)
    df = df.rename(columns={'Usage': 'usage', 'Emotion': 'label'})
    df['pixels'] = df['pixels'].apply(lambda x:
                                      np.fromstring(x, sep=' ', dtype=int))
    return df


if __name__ == '__main__':

    fer_data = '/Users/chris/Downloads/fer2013/fer2013.csv'
    df = _load_original_data_into_df(fer_data)
    this_path = os.path.abspath(__file__)
    this_dir, _ = os.path.split(this_path)
    df_path = os.path.join(this_dir, 'data', 'django_expression.pkl')
    df.to_pickle(df_path)
    # df.to_csv('/Users/chris/tmp/fer_processed.csv')
    # con = get_connection()
    # bucket = con.get_bucket('cmgreen210-emotions')
    # upload_big_file('/Users/chris/tmp/fer_processed.csv',
    #                 bucket)
