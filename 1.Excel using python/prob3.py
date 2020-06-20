from prob1 import Spreadsheet
import pickle

class PermanentSpreadsheet(Spreadsheet):
    def export_sheet(self, filename):
        if not isinstance(filename, str):
            raise TypeError('file name %s is not str type' % filename)
        with open(filename, 'wb') as f:
            pickle.dump(self.sheet, f)

    def import_sheet(self, filename):
        if not isinstance(filename, str):
            raise TypeError('file name %s is not str type' % filename)
        with open(filename, 'rb') as f:
            self.sheet = pickle.load(f)
spds08@login:~/HW2$ cat prob3.py
from prob1 import Spreadsheet

class Cell:
    def __init__(self,type,value)->None:
        self.type=type
        self.value = value
        self.function=None
        self.function_list=None
        
        
    def __str__(self):
        return str(self.value)
    
class SmartSpreadsheet(Spreadsheet):

    def set_function(self, operand_idx, f, idx):
            self.function[self.parse_idx(operand_idx)]=f
            self.function_list[self.parse_idx(operand_idx)]=self.parse_idx(idx)
            fnc=f
            op_val=self.sheet.get(self.parse_idx(idx))
            self.sheet[self.parse_idx(operand_idx)] = Cell(type(fnc(op_val.value)),fnc(op_val.value))
    
    def set_value(self, idx, value):
        if self.parse_idx(idx) in self.function_list:
            del(self.function[self.parse_idx(idx)])
            del(self.function_list[self.parse_idx(idx)])
        self.validate_value(value)
        self.sheet[self.parse_idx(idx)] = Cell(type(value), value)
        
    def get_value(self,idx):
            if self.parse_idx(idx) in self.function_list:
                cell_name=self.function_list.get(self.parse_idx(idx))
                if cell_name:
                    fnc=self.function.get(self.parse_idx(idx))
                    return fnc(self.sheet.get(cell_name).value)
            else:
                    cell = self.sheet.get(self.parse_idx(idx))
                    if cell:
                        return cell.value

            
