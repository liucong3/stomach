

text = '''
|    0     13153    C   python                                         149MiB |
|    0     13374    C   python                                       10479MiB |
|    0     14333    C   python                                         343MiB |
|    0     14881    C   python                                         195MiB |
|    1     13153    C   python                                         149MiB |
|    1     13374    C   python                                       10479MiB |
|    1     14333    C   python                                         343MiB |
|    1     14881    C   python                                         195MiB |
|    2     13153    C   python                                         149MiB |
|    2     13374    C   python                                       10479MiB |
|    2     14333    C   python                                         343MiB |
|    2     14881    C   python                                         195MiB |
|    3     13153    C   python                                           8MiB |
|    3     13374    C   python                                       10613MiB |
|    3     14333    C   python                                         351MiB |
|    3     14881    C   python                                         195MiB |
'''

import os
lines = text.split('\n')
for line in lines:
	data = line.split()
	if len(data) < 3: continue
	proc = data[2]
	os.system('kill ' + proc)
	