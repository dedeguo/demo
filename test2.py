#测试文件分隔符

file_path = 'dict/cement_term_dictionary.txt'
file = open(file_path, encoding='utf-8')
lines = file.read().splitlines()
for line in lines:
    print(line)
    print(line.split('\t'))