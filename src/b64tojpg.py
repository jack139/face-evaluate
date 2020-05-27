# -*- coding: utf-8 -*-

import os, sys
import base64

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("usage: python3 %s <base64 file>" % sys.argv[0])
        sys.exit(2)

    base64_file = sys.argv[1]

    file_name, file_ext = os.path.split(base64_file) 

    with open(base64_file, 'rb') as f1, \
        open(file_name+'.jpg', 'wb') as f2:
        f2.write(base64.b64decode(f1.read()))


