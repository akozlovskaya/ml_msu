def check(S, file_name):
    with open(file_name, 'w') as f:
        wds = S.lower().split(' ')
        wds.sort()
        d = {}
        for word in wds:
            if (word in d.keys()):
                d[word] += 1
            else:
                d[word] = 1
        for val in d.items():
            f.write(val[0] + ' ' + str(val[1]) + '\n')