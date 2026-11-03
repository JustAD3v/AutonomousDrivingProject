from Utils.Results import _avg, _std

class ResultsTable:
    """ TODO: implement result visualization for report """
    def __init__(self, row_keys=("model",), column_keys=("attack", "attack_bound"),
                 value_key="return_mean", std_key="return_std", title="Results"):
        self.row_keys = tuple(row_keys)
        self.column_keys = tuple(column_keys)
        self.value_key = value_key
        self.std_key = std_key
        self.title = title
        self.cells = {}
        self.rows = []
        self.columns = []
