import toml
import os
os.chdir('/openbayes/home/LMTDE/dev/data/toml_files')
cnf = toml.load('default_conf_new.toml')
print(cnf)