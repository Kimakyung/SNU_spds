class Cell:
    def __init__(self,type,value)->None:
        self.type=type
        self.value = value
        self.function=None
        self.function_list=None
        
        
    def __str__(self):
        return str(self.value)
    
class Spreadsheet(object):
    allowed_value_types = (int, bool, str)
    idx_pattern = '([a-zA-Z]+)(\d+)'

   
    def __init__(self):
        self.cell_format = '%-10s'
        self.allowed_columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        self.allowed_rows = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        self.sheet=dict()
        self.function = dict()
        self.function_list=dict()
        
    def parse_idx(self,idx):
        global column
        global row
        try:
            if (idx[0] in [chr(i) for i in range(ord('a'),ord('z')+1)] + [chr(i) for i in range(ord('A'),ord('Z')+1)]) and (int(idx[1]) in range(1,11)) and (len(idx)==2):
                column,row=idx
        except Exception as e:
            raise TypeError('%s is not valid index' % idx) from e

        if column.upper() not in self.allowed_columns or row not in self.allowed_rows:
            raise IndexError('%s is out of index' % idx)
        return column.upper(), row
    
    def validate_value(self, value):
        if not isinstance(value, self.allowed_value_types):
            raise TypeError('%s is not allowed. %s are allowed'
                            % (type(value), self.allowed_value_types))

    def set_value(self, idx, value):
        self.validate_value(value)
        self.sheet[self.parse_idx(idx)] = Cell(type(value), value)

    def get_value(self, idx):
        cell = self.sheet.get(self.parse_idx(idx))
        if cell:
            return cell.value

    def __str__(self):
        lines = []
        for r in self.allowed_rows:
            line = []
            for c in self.allowed_columns:
                value = self.get_value('%s%s' % (c, r)) or ''
                line.append(self.cell_format % value)
            lines.append(','.join(line))
        return '\n'.join(lines)
