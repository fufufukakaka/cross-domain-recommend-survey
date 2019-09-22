import scipy.sparse as sps
import numpy as np


class IncrementalSparseMatrix_ListBased(object):

    def __init__(self, auto_create_col_mapper = False, auto_create_row_mapper = False, n_rows = None, n_cols = None):

        super(IncrementalSparseMatrix_ListBased, self).__init__()

        self._row_list = []
        self._col_list = []
        self._data_list = []

        self._n_rows = n_rows
        self._n_cols = n_cols
        self._auto_create_column_mapper = auto_create_col_mapper
        self._auto_create_row_mapper = auto_create_row_mapper

        if self._auto_create_column_mapper:
            self._column_original_ID_to_index = {}

        if self._auto_create_row_mapper:
            self._row_original_ID_to_index = {}


    def add_data_lists(self, row_list_to_add, col_list_to_add, data_list_to_add):

        assert len(row_list_to_add) == len(col_list_to_add) and len(row_list_to_add) == len(data_list_to_add),\
            "IncrementalSparseMatrix: element lists must have different length"


        col_list_index = [self._get_column_index(column_id) for column_id in col_list_to_add]
        row_list_index = [self._get_row_index(row_id) for row_id in row_list_to_add]

        self._row_list.extend(row_list_index)
        self._col_list.extend(col_list_index)
        self._data_list.extend(data_list_to_add)




    def add_single_row(self, row_id, col_list, data = 1.0):

        n_elements = len(col_list)

        col_list_index = [self._get_column_index(column_id) for column_id in col_list]
        row_index = self._get_row_index(row_id)

        self._row_list.extend([row_index] * n_elements)
        self._col_list.extend(col_list_index)
        self._data_list.extend([data] * n_elements)



    def get_column_token_to_id_mapper(self):

        if self._auto_create_column_mapper:
            return self._column_original_ID_to_index.copy()



        dummy_column_original_ID_to_index = {}

        for col in range(self._n_cols):
            dummy_column_original_ID_to_index[col] = col

        return dummy_column_original_ID_to_index



    def get_row_token_to_id_mapper(self):

        if self._auto_create_row_mapper:
            return self._row_original_ID_to_index.copy()



        dummy_row_original_ID_to_index = {}

        for row in range(self._n_rows):
            dummy_row_original_ID_to_index[row] = row

        return dummy_row_original_ID_to_index



    def _get_column_index(self, column_id):

        if not self._auto_create_column_mapper:
            column_index = column_id

        else:

            if column_id in self._column_original_ID_to_index:
                column_index = self._column_original_ID_to_index[column_id]

            else:
                column_index = len(self._column_original_ID_to_index)
                self._column_original_ID_to_index[column_id] = column_index

        return column_index


    def _get_row_index(self, row_id):

        if not self._auto_create_row_mapper:
            row_index = row_id

        else:

            if row_id in self._row_original_ID_to_index:
                row_index = self._row_original_ID_to_index[row_id]

            else:
                row_index = len(self._row_original_ID_to_index)
                self._row_original_ID_to_index[row_id] = row_index

        return row_index


    def get_nnz(self):
        return len(self._row_list)



    def get_SparseMatrix(self):

        if self._n_rows is None:
            self._n_rows = max(self._row_list) + 1

        if self._n_cols is None:
            self._n_cols = max(self._col_list) + 1

        shape = (self._n_rows, self._n_cols)

        sparseMatrix = sps.csr_matrix((self._data_list, (self._row_list, self._col_list)), shape=shape)
        sparseMatrix.eliminate_zeros()


        return sparseMatrix





import numpy as np



class IncrementalSparseMatrix(IncrementalSparseMatrix_ListBased):

    def __init__(self, auto_create_col_mapper = False, auto_create_row_mapper = False, n_rows = None, n_cols = None, dtype = np.float64):

        super(IncrementalSparseMatrix, self).__init__(auto_create_col_mapper = auto_create_col_mapper,
                                                             auto_create_row_mapper = auto_create_row_mapper,
                                                             n_rows = n_rows,
                                                             n_cols = n_cols)

        self._dataBlock = 10000000
        self._next_cell_pointer = 0

        self._dtype_data = dtype
        self._dtype_coordinates = np.uint32
        self._max_value_of_coordinate_dtype = np.iinfo(self._dtype_coordinates).max

        self._row_array = np.zeros(self._dataBlock, dtype=self._dtype_coordinates)
        self._col_array = np.zeros(self._dataBlock, dtype=self._dtype_coordinates)
        self._data_array = np.zeros(self._dataBlock, dtype=self._dtype_data)


    def get_nnz(self):
        return self._next_cell_pointer


    def add_data_lists(self, row_list_to_add, col_list_to_add, data_list_to_add):

        assert len(row_list_to_add) == len(col_list_to_add) and len(row_list_to_add) == len(data_list_to_add),\
            "IncrementalSparseMatrix: element lists must have the same length"

        for data_point_index in range(len(row_list_to_add)):

            if self._next_cell_pointer == len(self._row_array):
                self._row_array = np.concatenate((self._row_array, np.zeros(self._dataBlock, dtype=self._dtype_coordinates)))
                self._col_array = np.concatenate((self._col_array, np.zeros(self._dataBlock, dtype=self._dtype_coordinates)))
                self._data_array = np.concatenate((self._data_array, np.zeros(self._dataBlock, dtype=self._dtype_data)))


            row_index = self._get_row_index(row_list_to_add[data_point_index])
            col_index = self._get_column_index(col_list_to_add[data_point_index])

            self._row_array[self._next_cell_pointer] = row_index
            self._col_array[self._next_cell_pointer] = col_index
            self._data_array[self._next_cell_pointer] = data_list_to_add[data_point_index]

            self._next_cell_pointer += 1




    def add_single_row(self, row_index, col_list, data = 1.0):

        n_elements = len(col_list)

        self.add_data_lists([row_index] * n_elements,
                            col_list,
                            [data] * n_elements)





    def get_SparseMatrix(self):

        if self._n_rows is None:
            self._n_rows = self._row_array.max() + 1

        if self._n_cols is None:
            self._n_cols = self._col_array.max() + 1

        shape = (self._n_rows, self._n_cols)

        sparseMatrix = sps.csr_matrix((self._data_array[:self._next_cell_pointer],
                                       (self._row_array[:self._next_cell_pointer], self._col_array[:self._next_cell_pointer])),
                                      shape=shape,
                                      dtype=self._dtype_data)

        sparseMatrix.eliminate_zeros()


        return sparseMatrix

class IncrementalSparseMatrix_FilterIDs(IncrementalSparseMatrix):
    """
    This class builds an IncrementalSparseMatrix allowing to constrain the row and column IDs that will be added
    It is useful, for example, when
    """

    def __init__(self, preinitialized_col_mapper = None, preinitialized_row_mapper = None,
                 on_new_col = "add", on_new_row = "add", dtype = np.float64):
        """
        Possible behaviour is:
        - Automatically add new ids:    if_new_col = "add" and predefined_col_mapper = None or predefined_col_mapper = {dict}
        - Ignore new ids                if_new_col = "ignore" and predefined_col_mapper = {dict}
        :param preinitialized_col_mapper:
        :param preinitialized_row_mapper:
        :param on_new_col:
        :param on_new_row:
        :param n_rows:
        :param n_cols:
        """

        super(IncrementalSparseMatrix_FilterIDs, self).__init__(dtype = dtype)

        self._row_list = []
        self._col_list = []
        self._data_list = []

        assert on_new_col in ["add", "ignore"], "IncrementalSparseMatrix: if_new_col value not recognized, allowed values are 'add', 'ignore', provided was '{}'".format(on_new_col)
        assert on_new_row in ["add", "ignore"], "IncrementalSparseMatrix: if_new_row value not recognized, allowed values are 'add', 'ignore', provided was '{}'".format(on_new_row)

        if on_new_col == "add":
            assert preinitialized_col_mapper is None or isinstance(preinitialized_col_mapper, dict), "IncrementalSparseMatrix: if on_new_col is 'add' then preinitialized_col_mapper must be either 'None' or contain a dictionary"

        if on_new_row == "add":
            assert preinitialized_row_mapper is None or isinstance(preinitialized_row_mapper, dict), "IncrementalSparseMatrix: if on_new_row is 'add' then preinitialized_row_mapper must be either 'None' or contain a dictionary"

        if on_new_col == "ignore":
            assert isinstance(preinitialized_col_mapper, dict), "IncrementalSparseMatrix: if on_new_col is 'ignore' then preinitialized_col_mapper must be a dictionary"

        if on_new_row == "ignore":
            assert isinstance(preinitialized_row_mapper, dict), "IncrementalSparseMatrix: if on_new_row is 'ignore' then preinitialized_row_mapper must be a dictionary"


        self._on_new_col_add_flag = on_new_col == "add"
        self._on_new_row_add_flag = on_new_row == "add"

        self._auto_create_row_mapper = True
        self._auto_create_column_mapper = True


        if preinitialized_col_mapper is None:
            self._column_original_ID_to_index = {}
        else:
            self._column_original_ID_to_index = preinitialized_col_mapper.copy()

        if preinitialized_row_mapper is None:
            self._row_original_ID_to_index = {}
        else:
            self._row_original_ID_to_index = preinitialized_row_mapper.copy()




    def _get_column_index(self, column_id):

        if column_id in self._column_original_ID_to_index:
            column_index = self._column_original_ID_to_index[column_id]

        elif self._on_new_col_add_flag:
            column_index = len(self._column_original_ID_to_index)
            self._column_original_ID_to_index[column_id] = column_index

        else:
            column_index = None

        return column_index




    def _get_row_index(self, row_id):

        if row_id in self._row_original_ID_to_index:
            row_index = self._row_original_ID_to_index[row_id]

        elif self._on_new_row_add_flag:
            row_index = len(self._row_original_ID_to_index)
            self._row_original_ID_to_index[row_id] = row_index

        else:
            row_index = None

        return row_index




    def add_data_lists(self, row_list_to_add, col_list_to_add, data_list_to_add):

        assert len(row_list_to_add) == len(col_list_to_add) and len(row_list_to_add) == len(data_list_to_add),\
            "IncrementalSparseMatrix: element lists must have different length"


        for data_point_index in range(len(row_list_to_add)):

            if self._next_cell_pointer == len(self._row_array):
                self._row_array = np.concatenate((self._row_array, np.zeros(self._dataBlock, dtype=self._dtype_coordinates)))
                self._col_array = np.concatenate((self._col_array, np.zeros(self._dataBlock, dtype=self._dtype_coordinates)))
                self._data_array = np.concatenate((self._data_array, np.zeros(self._dataBlock, dtype=self._dtype_data)))


            row_index = self._get_row_index(row_list_to_add[data_point_index])
            col_index = self._get_column_index(col_list_to_add[data_point_index])


            if row_index is not None and col_index is not None:

                self._row_array[self._next_cell_pointer] = row_index
                self._col_array[self._next_cell_pointer] = col_index
                self._data_array[self._next_cell_pointer] = data_list_to_add[data_point_index]

                self._next_cell_pointer += 1



    def get_SparseMatrix(self):

        # Set fixed dimension len to ensure that the matrix is not smaller than the number of entries in the dictionary
        self._n_rows = len(self._row_original_ID_to_index)
        self._n_cols = len(self._column_original_ID_to_index)


        return super(IncrementalSparseMatrix_FilterIDs, self).get_SparseMatrix()

def split_train_validation_leave_one_out_user_wise(URM_train, verbose=True, at_least_n_train_items=0):

    num_users, num_items = URM_train.shape

    URM_train_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items)
    URM_validation_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items)

    count_train = 0
    count_validation = 0
    for user_id in range(URM_train.shape[0]):

        start_pos = URM_train.indptr[user_id]
        end_pos = URM_train.indptr[user_id+1]


        user_profile_items = URM_train.indices[start_pos:end_pos]
        user_profile_ratings = URM_train.data[start_pos:end_pos]
        user_profile_length = len(user_profile_items)

        n_train_items = user_profile_length

        if n_train_items > at_least_n_train_items:
            n_train_items -= 1

        indices_for_sampling = np.arange(0, user_profile_length, dtype=np.int)
        np.random.shuffle(indices_for_sampling)

        train_items = user_profile_items[indices_for_sampling[0:n_train_items]]
        train_ratings = user_profile_ratings[indices_for_sampling[0:n_train_items]]

        validation_items = user_profile_items[indices_for_sampling[n_train_items:]]
        validation_ratings = user_profile_ratings[indices_for_sampling[n_train_items:]]

        if len(train_items) == 0:
            if verbose: print("User {} has 0 train items".format(user_id))
            count_train += 1

        if len(validation_items) == 0:
            if verbose: print("User {} has 0 validation items".format(user_id))
            count_validation += 1

        URM_train_builder.add_data_lists([user_id]*len(train_items), train_items, train_ratings)
        URM_validation_builder.add_data_lists([user_id]*len(validation_items), validation_items, validation_ratings)

    if count_train>0:
        print("{} users with 0 train items".format(count_train))
    if count_validation>0:
        print("{} users with 0 validation items".format(count_validation))


    URM_train = URM_train_builder.get_SparseMatrix()
    URM_validation = URM_validation_builder.get_SparseMatrix()


    return URM_train, URM_validation

def ndcg(ranked_list, pos_items, relevance=None, at=None):

    if relevance is None:
        relevance = np.ones_like(pos_items)
    assert len(relevance) == pos_items.shape[0]

    # Create a dictionary associating item_id to its relevance
    # it2rel[item] -> relevance[item]
    it2rel = {it: r for it, r in zip(pos_items, relevance)}

    # Creates array of length "at" with the relevance associated to the item in that position
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in ranked_list[:at]], dtype=np.float32)

    # IDCG has all relevances to 1, up to the number of items in the test set
    ideal_dcg = dcg(np.sort(relevance)[::-1])

    # DCG uses the relevance of the recommended items
    rank_dcg = dcg(rank_scores)

    if rank_dcg == 0.0:
        return 0.0

    ndcg_ = rank_dcg / ideal_dcg
    # assert 0 <= ndcg_ <= 1, (rank_dcg, ideal_dcg, ndcg_)
    return ndcg_


def dcg(scores):
    return np.sum(np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
                  dtype=np.float32)
