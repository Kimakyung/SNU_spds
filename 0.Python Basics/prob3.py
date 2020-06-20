def comment_remover(s):
    return print('\n'.join(l[:l.find('#')] for l in s.split('\n')))
